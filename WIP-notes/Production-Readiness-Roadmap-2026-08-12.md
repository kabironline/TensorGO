# nanograd — Production-Readiness Roadmap

> Authored 2026-08-12 after a full re-audit of `tensor/`, `storage/`, `nn/`, `optim/`,
> `backend/`, `internal/backend/{cpu,cuda}`, kernels, and every test.
> **Supersedes** [Engine-Redesign-Roadmap-2026-06-22.md](./Engine-Redesign-Roadmap-2026-06-22.md).
> Working mode: Claude is architect & reviewer; **you write every line.**

---

## Why this document replaces the old roadmap

The June roadmap set out to fix three things: GPU leaks, float32-only, and serialization.
Those are real. But it assumed the engine underneath was *correct* and merely needed a
better memory model. The re-audit says otherwise.

**The engine computes wrong gradients, today, on the CPU, in its most-used operations —
and no test can see it.**

That changes the order of everything. You cannot safely migrate memory ownership or add a
second dtype on top of an engine whose `MatMul` backward is wrong, because every refactor
after that point is unverifiable: you'd have no way to tell a migration regression from a
pre-existing bug. Correctness and a test harness that can *prove* correctness have to come
first. Multi-dtype — the original Phase 1 goal — is now near the end, and that is the right
place for it.

The old roadmap's core technical insight (Storage as a typed buffer; asymmetric CPU/GPU
ownership; Scope for GPU lifetime) is **sound and retained**. What changes is the sequencing,
and the addition of five workstreams it didn't cover at all: buildability, error handling,
device safety, module composability, and test infrastructure.

### What is genuinely done and worth keeping

- `storage` package, `Buffer` interface, `hostBuffer`/`gpuBuffer` split — good design, keep.
- `backend.StorageManager` — right API, still correctly unadopted.
- CUDA power-of-2 caching pool (`memory_pool.go`) — sound allocator; needs stats and a
  use-after-free fix, not a rewrite.
- Define-by-run autograd topology in `engine.go` — the graph walk is correct; its memory
  teardown and nil-grad handling are not.
- Adam's math (`cpu/ops_optimizer.go:25-41`) — bias correction and epsilon placement are
  PyTorch-equivalent and verified by the one good test in the repo.

---

## The six things that make this not-production-ready

Ordered by how badly they'd bite a real user.

**1. It computes wrong gradients silently.** Not crashes — wrong numbers, no diagnostic.
Confirmed by reading the code:

| Bug | Location | Effect |
|---|---|---|
| `Add` fast path ignores strides/offset | `tensor/ops_matrix.go:10-31` | `aT.Add(b)` adds the *untransposed* base buffer. Sizes the op by `len(a.Data())`, not `TotalSize(a.Shape)`. |
| `MatMul` passes `Strides[0]` as the BLAS leading dimension with no contiguity check | `tensor/ops_matrix.go:188,193,199` | `blas32.General{Stride:}` requires the *inner* stride to be 1, but `Strides[0]` is the *outer* stride. For a transposed 2-D view `Strides==[1,m]` ⇒ `lda==1`. **CPU: gonum's `Stride >= Cols` check rejects it — `panic: blas: bad leading dimension of A`, so a transposed matmul simply does not work.** **CUDA: cuBLAS does not validate `lda`, so the identical call reads the wrong memory and returns garbage silently** (`cuda/ops_matrix.go:27`). Confirmed by test 2026-08-13. `MatMulTransA`/`MatMulTransB` share the bug and run on every backward pass — on CUDA they also discard their return codes (`ops_matrix.go:41,53`). |
| `Slice` applies its offset twice | `tensor/ops_matrix.go:409,413` | Storage is resliced to `[offset:]` *and* `Offset` is set. Every read lands at `base[2*offset]`. |
| `Slice` backward accumulates into an unzeroed pooled buffer | `tensor/ops_matrix.go:426` | Uses `pools.GetBuffer`, not `GetZeroedBuffer`, then `+=`. Gradient corruption that varies with pool reuse. |
| `Inverse` backward never reads the incoming gradient | `tensor/ops_matrix.go:332-333` | `mGradOut` is built from `resultData`, not `out.Grad()`. Gradient is independent of `dL/dout`. |
| `MatVecMul`/`VecMatMul` drop the vector's gradient | `tensor/ops_matrix.go:262-271,284-293` | Hand-rolled views with no `Parents`/`Backward`; the vector never trains. |
| `Reshape` on a non-contiguous tensor relabels garbage | `tensor/ops_transpose.go:102-143` | No contiguity check. |
| Two contradictory grad-buffer sizing rules | `tensor/tensor.go:365` vs `:377` | `ensureGrad` uses storage length, `AllocGrad` uses `TotalSize(Shape)`. Broadcast views under-allocate. |
| `AccumulateGrad` stores grads in *logical* order on one path and *physical* on another | `tensor/tensor.go:240` vs `:266` | Path chosen by an incidental length comparison. Slice-then-transpose is wrong. |
| `BroadcastAddOp`/`Sub`/`Mul`/`DivOp` set `Parents` but no `Backward` | `tensor/ops_broadcast.go:104-188` | Exported. Backprop silently yields zero. `tests/tensor_test.go:117` calls one directly. |
| `cuda_contiguous` arg order swapped vs its header | `cuda/ops_tensor.go:31-32` vs `kernels/tensor/ops_tensor.h:12` | Header is `(ndim, offset, total)`; Go passes `(ndim, total, offset)`. Kernel early-returns, and the caller hands back an unzeroed pool block as the "contiguous copy". |
| `[]int` (8 bytes) passed as `*C.int` (4 bytes) | `cuda/ops_tensor.go:28-29` | Device sees `shape=[2,0]`; kernel does `idx % 0`. |

**2. Nobody can build it.** `go build ./...` fails. `internal/backend/cuda` has **zero build
tags**, but `op_stubs.go` and `storage.go` are non-cgo files, so with `CGO_ENABLED=0` they
survive and reference `CUDABackend`/`MemoryPool`, which don't exist. Hardcoded
`/usr/local/cuda/lib64` and `/usr/lib/wsl/lib` mean it can't build on Windows (this repo's own
platform) or macOS even *with* a GPU. `go get github.com/kabironline/nanograd` is broken for
every user on earth who lacks that exact Linux/WSL layout.

**3. No test can catch a wrong gradient.** There is **no finite-difference gradient checker
anywhere in the repo.** Every backward is checked against hand-computed constants, which is
precisely why the bugs above survived. Tests live in sibling packages (`tensor/tests`,
`internal/backend/cpu/test`), so `go test -cover` reports **0.0%** for every package that
contains logic; measured with `-coverpkg`, the whole suite reaches ~33%. `nn/tests` — home of
the only end-to-end test — won't build without CUDA. Every `example/` test fails to compile.
The MNIST >95% and CIFAR >20% accuracy gates have not run in a long time. No CI exists.

**4. Save/Load is broken in four independent ways.** Loaded models can't train
(`nn/sequential.go:133` → `tensor.NewTensor` with no parents ⇒ `RequiresGrad=false` ⇒
`AccumulateGrad` early-returns ⇒ silent no-op training). Conv/Pool/Flatten layers are
**silently dropped** by `LoadModuleAt` (`nn/sequential.go:158` returns `nil,nil`, and `Load`
skips nils) — every CNN loads back truncated, with `err == nil`. Saving a GPU model ranges
over a device pointer from the host (`nn/linear.go:49`). Optimizer state is never saved, so
Adam resume is a hard reset with a 10× bias-corrected first step into converged weights.

**5. The fake slice makes device memory indistinguishable from host memory.**
`cuda/ops_memory.go:107` returns `unsafe.Slice((*float32)(devPtr), n)` as a plain
`[]float32`. Seven unguarded host-side sites index it and would segfault on GPU —
`tensor.go:132` (`Eye`), `stride.go:35,40` (`At`/`Set`), `ops_matrix.go:309`,
`ops_reduction.go:24,55`, `nn/linear.go:49,65`, `nn/conv/conv2d.go:135,151`. `storage.go`
already models the right answer (`gpuBuffer.Bytes()` panics); `Allocate` deliberately defeats
it. **There is no working GPU→CPU path at all**: `Tensor.To` falls back to
`dev.Copy(hostSlice, deviceSlice)` — a host memcpy from a device pointer.

**6. 87 panics vs 18 error returns.** Shape mismatches, singular matrices, device OOM, and
"no backend available" all abort the process. A library that panics on a bad input shape
cannot be embedded in a server. Compounding it, the backend registry
(`backend/backend.go:359-367`) is an unsynchronized package-level map — concurrent
`RegisterBackend` + `AutoSelectBackend` is a fatal Go map race, and `tensor.NewTensor` calls
`AutoSelectBackend` on every construction.

---

## Phase plan

Each phase ends with `go build ./...` clean on **Linux, Windows, and macOS** (CPU-only) and the
full suite green. Phases are ordered so that every later phase is verifiable by gates the
earlier ones installed.

| Phase | Theme | Why here |
|---|---|---|
| **P0** ✅ | Build, test infrastructure, CI | Nothing else is verifiable until the code builds and gradients can be checked |
| **P1** | Autograd correctness (CPU) | Fix wrong math while the harness can prove it |
| **P2** | Ownership & lifetime | Now safe: any numeric change is a memory bug, not a math bug |
| **P3** | Device safety (kill the fake slice) | Needs P2's ownership model to have somewhere to put device handles |
| **P4** | Errors & API surface | Breaking API changes land once, after the semantics are settled |
| **P5** | Serialization v2 | Needs P3 (GPU staging) and P4 (module composability) |
| **P6** | Multi-dtype | The original goal — now provable and safe to land |
| **P7** | Performance & release | Optimize only what is correct |

---

### P0 — Make it buildable and make correctness observable ✅ **DONE 2026-08-13**

**Nothing in this phase changes behavior.** It builds the instruments.

**Completion notes.** All eight items landed. Verified in five configurations:
`CGO_ENABLED=0 go build ./...`, `CGO_ENABLED=0 go test -run=NONE ./...`,
`go build -tags cuda ./...`, `go test -tags cuda -run=NONE ./...`, `go vet`,
`gofmt -l`, and `go test -race`, on both Windows and WSL.

The gradcheck harness found **three** of the predicted P1 bugs on its first run,
now marked with skips in `tensor/gradcheck_test.go` that name the fix:

| Check | Symptom | Bug |
|---|---|---|
| `views/add-on-transposed-view` | analytic grads are a *transposed permutation* of the true ones | `Add` fast path ignores strides |
| `views/slice` | gradient lands 6 elements past where it belongs | `Slice` applies offset twice |
| `views/slice-then-transpose` | panic, index out of range | `ensureGrad`/`AllocGrad` size disagreement |

34 of 37 checks pass, including the whole matmul family and every activation.

**Two design lessons worth carrying forward:**

1. **A plain `Sum()` loss cannot detect a permutation bug.** Sum is
   permutation-invariant, so `d(sum)/dx` is 1 for every element regardless of the
   order the op emits them in — the `Add`-on-transposed-view bug passed until the
   harness switched to a *position-weighted* sum (`sum(out * w)`, `w` constant and
   distinct per position). Any future reduction used as a gradcheck loss must
   break that symmetry.
2. **`go test -run <pattern-that-matches-nothing>` exits 0.** The first canary run
   reported `PASS` having executed zero tests. The CI job asserts that gradchecks
   actually ran.

1. **Build tags for CUDA.** Split every cgo file with `//go:build cuda` and add a
   `cuda_stub.go` with `//go:build !cuda` providing a `NewCUDABackend` that returns
   `(nil, ErrCUDAUnavailable)`. Move `op_stubs.go` and `storage.go` behind the same tag.
   Gate: `CGO_ENABLED=0 go build ./...` passes on all three OSes.
2. **Make CGO flags discoverable, not hardcoded.** `CUDA_PATH` env var with a
   platform-appropriate default; drop `-lm`/`-lstdc++` on Windows. **Rename
   `kernels/libcuda.a`** — it collides with the NVIDIA driver stub, and five files carry a
   duplicate `-lcuda` that `ld` may resolve to your archive instead of the driver.
3. **Fix the compile breaks the Storage migration left behind**: all four `example/` tests
   (`.Data` → `.Data()`; the GPU test also builds a composite literal with now-unexported
   fields) and `nn/tests/backend_selection_test.go`'s unconditional cuda import.
4. **Move tests into the packages they test.** `tensor/tests/` → `tensor/`,
   `internal/backend/cpu/test/` → `internal/backend/cpu/`. This is what makes coverage real
   and unexported functions (`ensureGrad`, `clearGraphHelper`, `getIndex`, `broadcastShapes`)
   reachable — all currently at zero coverage.
5. **Build the finite-difference gradient checker.** The single highest-value artifact in
   this roadmap:
   ```
   gradcheck.Check(t, fn func([]*Tensor) *Tensor, inputs []*Tensor, tol float64)
   ```
   Central differences, `f(x+h) - f(x-h) / 2h`, compared against the analytic backward.
   Use `float64` accumulation for the numeric side. Run it over *every* op, and critically
   over **views**: transposed, sliced, offset, broadcast, and reshaped inputs.
6. **Seedable RNG.** `cpu/ops_random.go:9` uses the unseeded global `math/rand`; weights
   differ every run, so any accuracy gate is inherently flaky and failures are unreproducible.
   Add `Seed(uint64)` to `RandomOps`.
7. **CI** (`.github/workflows`): build matrix ubuntu/windows/macos × `CGO_ENABLED={0,1}`,
   `go vet`, `go test -race`, coverage floor. CUDA jobs build-only (no GPU runner).
8. **Repo hygiene**: `README.md`, `Makefile`, delete the empty `dtype/` directory, untrack
   `mem.out`/`trace.out`/`kernels/libcuda.a`.

**Gate:** `go build ./...` on 3 OSes; gradcheck harness exists and is wired into CI;
`go test -race ./...` green.

**Expect the gradcheck to fail loudly on first run.** That is the phase succeeding.

---

### P1 — Autograd correctness on CPU

Work the P1 table from the summary above. Two structural decisions drive most of it, and you
should make them explicitly before touching individual ops:

**Decision 1 — one canonical gradient layout.** A tensor's `grad` is *always* logical-order,
contiguous, and sized `TotalSize(Shape)`. No exceptions. This kills the `ensureGrad`/`AllocGrad`
divergence, kills `AccumulateGrad`'s two-conventions bug, and makes the broadcast-view
under-allocation impossible to express.

**Decision 2 — derive `contiguous`, never store it.** It is a manually-maintained bool that
`Transpose`, `Reshape`, `BroadcastTo`, `Slice`, and the hand-rolled 2-D views all forget to
set. Make it `strides == ComputeStrides(shape) && Offset == 0`, computed on demand. A
contiguous `Reshape` currently labels itself non-contiguous, forcing a redundant copy on every
downstream op and pushing `AccumulateGrad` off its fast path.

Then, in order:
1. `engine.go:56` — skip `Backward()` when `current.grad == nil`. A `RequiresGrad=false` node
   can be an ancestor of a `RequiresGrad=true` node; `ReLU`'s backward then indexes a nil
   slice. This is a live crash, not a theoretical one.
2. `Add` and `MatMul` — route through `Contiguous()` like every other binary op does, and size
   by `TotalSize(Shape)`.
3. `Slice` — pick reslicing *or* `Offset`, not both; use `GetZeroedBuffer` in its backward.
4. `Inverse` backward — read `out.Grad()`; route through `AccumulateGrad` instead of writing
   `a.Grad()` directly.
5. `MatVecMul`/`VecMatMul` — use `b.Reshape(...)` instead of hand-rolled views.
6. `Reshape` — copy or reject on non-contiguous input.
7. Unexport `BroadcastAddOp`/`Sub`/`Mul`/`DivOp`, or give them real backwards.
8. `Contiguous(t)` (`stride.go:59`) silently detaches from the graph — it's exported, so
   `tensor.Contiguous(x)` in user code is a gradient black hole. Unexport it or give it a
   copy-backward.
9. Add `Tensor.ZeroGrad`, `Detach`, `Clone`; `BackProp` should reject a non-scalar root
   instead of seeding all-ones.
10. Dedupe `ComputeStrides`/`computeStrides`; resolve `Contiguous` meaning both a package
    function returning `*Tensor` and a method returning `bool`.

**Gate:** gradcheck passes for every op × {contiguous, transposed, sliced, offset, broadcast,
reshaped} input. A test that calls `BackProp` twice. A test mixing `RequiresGrad=false` nodes.

---

### P2 — Ownership and lifetime

This is the old roadmap's P4, now with a concrete bug list and a real gate underneath it.

The central problem: **`Storage` has no owner.** `Transpose`, `Reshape`, `BroadcastTo`, and
`Slice` all set `data: t.data` *and* `Parents: []*Tensor{t}`. `clearGraphHelper`
(`tensor.go:321-362`) treats "has Parents" as "is an intermediate, free its data" — so a view
of a *parameter* gets classified as an intermediate and its buffer is freed, while the same
function's `else` branch explicitly promises not to free parameters. Two views of one buffer
make it a genuine double-free; on CUDA the second `memPool.Free` panics.

1. Add an explicit ownership bit to `Storage` (owner vs. view), or make views hold a
   reference to the owning `Storage` rather than aliasing the pointer. Do **not** add
   refcounting to CPU storage — Go's GC already handles that correctly, and the asymmetry
   decision from the old roadmap still holds.
2. Make `Storage.Free` idempotent (`storage.go:127` never nils `buf`).
3. Introduce the **`Scope`** (`dev.Step(func(s *Scope){...})` + `defer s.Release()`) for GPU
   transients. Delete `ClearGraph`, `ClearComputationGraph`, and the `runtime.SetFinalizer`
   teardown — three overlapping lifecycle methods with no stated contract.
4. `Tensor.To` must return a *new* tensor and free the source, not mutate the receiver in
   place while leaking the old buffer and desynchronizing every existing view.
5. Give `ReduceSumTo` an unambiguous ownership contract — it currently returns its input when
   shapes match, and callers free it inconsistently (`Div` never frees at all).
6. `Contiguous()` copies are allocated on every non-contiguous op and never freed;
   `ReLU`/`Log`/`Square` additionally *capture* the copy in their backward closure, pinning a
   duplicate of every activation for the graph's lifetime — and reading a stale snapshot if the
   optimizer updates the parameter before backward runs.
7. Adam's `M`/`V` are allocated from the device pool and never freed; no `Close()`. Also fix
   the device-ordering hazard: `M`/`V` bind to `p.Device` at construction, so
   `model.To(cuda)` *after* `NewAdam` silently feeds host buffers to `cuda_step_adam`.
8. **P0-from-the-old-roadmap: allocator `Stats()`** (`BytesInUse`/`BytesReserved`/`ActiveBlocks`)
   on both backends. Build it *first* in this phase — it is the gate for everything else here.

**Gate:** a training loop runs N steps with **no** `ClearGraph()` call and `Stats()` returns to
baseline. `go test -race` green.

---

### P3 — Device safety: kill the fake slice

`Allocate` must stop returning `[]float32` over device memory. This is the phase where
`StorageManager`/`gpuBuffer` finally get **adopted** — and, exactly as the June roadmap
concluded, it is coupled to converting the compute ops to `*Storage` signatures. That
conclusion was right; it just belongs after correctness and ownership, not before.

1. Convert `Backend` op signatures from `[]float32` to `*storage.Storage`, family by family
   (elementwise → matrix → reduction → activation → conv). Delete `MemoryManager`'s
   `[]float32` methods as each family lands.
2. Delete the `Data()`/`Grad()` shims — but note they have **write** dependencies
   (`optim/sgd.go:21` writes parameters through them), so they need `CopyFrom`/`Fill`
   replacements, not just deletion.
3. Fix the two CUDA kernel-boundary bugs: `cuda_contiguous`'s swapped `total`/`offset`, and
   the `[]int`→`*C.int` truncation. Build a `[]C.int` like `ops_reduction.go:43` already does.
4. `runtime.LockOSThread` around device-bound work. `cudaSetDevice` is thread-local; Go
   migrates goroutines across threads, so with `deviceID != 0` every pointer is foreign to the
   calling thread. This makes multi-GPU unusable and is a strong candidate for the
   intermittent `ToCPU copy failed: 1`.
5. Register the CUDA backend — `NewCUDABackend` is **never** passed to `RegisterBackend`, so
   `backends["cuda"]` can never hit. Also reconcile the name: `Base` is built as `"cuda:0"`
   while lookups use `"cuda"`.
6. Fix the global cuDNN workspace (`convolution.cu:22-51`): `cudaFree` + realloc with no
   stream sync while a conv may still be reading it — the clearest true async use-after-free
   in the tree — plus no mutex on a file-scope global.
7. Sync policy: five unary ops bypass the stream entirely and `cudaDeviceSynchronize` after
   every launch; `Copy` uses blocking `cudaMemcpy`; `ToDevice`/`WriteToDevice` issue async H2D
   from Go heap memory and return immediately (a cgo pointer-rule violation).
8. Fix `WorkerPool`'s single shared `WaitGroup` (`cpu/worker_pool.go:16`) — concurrent
   `Process` calls panic with "WaitGroup is reused before previous Wait has returned"; a
   `Close()` race can also drop a `Done()` and hang forever.
9. CPU/CUDA semantic divergence: `cpu/ops_dmas.go:46,53,60` iterate `len(out)` and ignore the
   `size` argument that CUDA honors.

**Gate:** `compute-sanitizer --tool memcheck` clean on the CUDA suite. Host code touching a
device buffer **panics** rather than segfaults. `TestCudaDMAS` passes (this is where the
long-standing `ToCPU copy failed: 1` should finally resolve — the prime suspects are #4 above
and P2's double-free, not the allocator, whose sizing and alignment I checked and found sound).

---

### P4 — Errors and API surface

The last phase that breaks API compatibility. Everything after this is additive.

1. **Draw the panic/error line.** Panic only for programmer error (nil receiver, internal
   invariant). Return `error` for everything a caller can hit with valid code: shape
   mismatch, device mismatch, dtype mismatch, singular matrix, OOM, no-backend-available.
   Realistically this means `Add`/`MatMul`/… gain `TryAdd`/`TryMatMul` variants, or the ops
   return `(*Tensor, error)` and you accept the ergonomic cost.
2. **Device and dtype mismatch are never checked at all** — `ops_matrix.go:12` uses
   `a.Device` to operate on `b.Data()`. Mixing a CPU and a CUDA tensor hands a device pointer
   to a CPU loop. Add the check.
3. **Mutex the backend registry**, or better: eliminate the global. A device should be
   scopable to a model or a call, not process-global — the tests already mutate global state
   with `defer` restore, which makes parallel tests unsafe.
4. **Split the `Backend` interface.** It is a 60-method monolith whose own doc comment claims
   it uses optional interfaces "to avoid forcing backends to implement everything" while
   embedding all 15 sub-interfaces in the required set — which makes the `SupportsConvolutions`
   / `SupportsMemoryTransfer` helpers below it dead code that always returns true. Required
   core: memory + elementwise + matmul. Everything else optional with a real fallback path.
5. **Make `Module` composable.** `Sequential` doesn't implement `Module` (its `Save` has a
   different signature), so `Sequential` can't nest — no ResNet blocks, no reusable
   sub-modules. Add `var _ Module = (*X)(nil)` assertions everywhere; their absence is why this
   and the dropped-conv-layer bug went unnoticed. Fix the value/pointer receiver split on
   `Sequential`. Add `Train()`/`Eval()` before Dropout/BatchNorm exist, not after.
6. Encapsulate `Tensor`'s exported mutable fields (`Parents`, `Strides`, `Shape`, `Backward`).
   Callers currently reassign `out.Parents` after construction as normal practice, so graph
   invariants are unenforceable.
7. Add the missing constructors: `Zeros`, `Ones`, `Full`, `Rand`, `Arange`, `Clone`, `Detach`,
   `String`.
8. `context.Context` on training-loop entry points for cancellation.

**Gate:** `go vet` clean; API documented; no global mutable state reachable from a hot path.

---

### P5 — Serialization v2

1. `RequiresGrad` must survive a round trip (one line; currently silently destroys all
   fine-tuning).
2. `LoadModuleAt` must handle Conv2D, MaxPool2D, and Flatten, and must **error** on an
   unrecognized key rather than returning `(nil, nil)` and truncating the model.
3. Serialize layer hyperparameters — `Conv2D.Stride`/`Padding` and all of `MaxPool2D`'s config
   are currently written nowhere, so even a fixed loader couldn't reconstruct them.
4. Replace the activation-as-dummy-1-element-tensor marker hack with real config in
   `__metadata__`; it can't represent a parameterized activation (LeakyReLU slope, GELU
   variant) at all.
5. Populate `__metadata__`: format version, framework tag, dtype/layout contract, architecture
   descriptor. It is currently passed `nil`. Reject checkpoints from an unknown version.
6. Hierarchical keys (`block.0.conv.weight`) to match P4's nestable modules; the flat
   `layer_%d.` scheme has no room for a hierarchy.
7. All tensors staged through `Storage.ToHostBytes()`/`CopyFromHostBytes()` — never a raw
   `Data()` range, which is what breaks GPU save today.
8. `StatefulOptimizer{StateDict/LoadStateDict}` for Adam `M`/`V`/`T`, plus LR, betas, step
   count, and RNG state.
9. Atomic write: temp file + `fsync` + `rename`. Today `os.WriteFile` overwrites in place, so
   a crash mid-write destroys the previous good checkpoint too.
10. Stream the write — it currently builds the whole blob in memory plus a full byte copy per
    tensor, ~3× model size peak RSS.

**Gate:** save→load→train reproduces the pre-save loss curve. A CNN round-trips with every
layer intact. Adam resumed at step N reproduces step N+1 exactly. A GPU model saves real
values.

---

### P6 — Multi-dtype

The original Phase 1, now landing on an engine that can prove it didn't break.

1. `switch a.DType()` dispatch in each backend method; generics inside leaf CPU kernels only.
   The runtime-tag decision (vs. `Tensor[T]` generics) remains correct — interface methods
   can't be generic, and a mixed-dtype `[]*Tensor`/Module/Adam is unrepresentable otherwise.
2. F64 on CPU first: `blas32.Gemm`/`blas64.Gemm` split. CUDA panics on non-F32.
3. Explicit `Cast` op. Never a silent CPU fallback — that is exactly the `op_stubs.go` pattern
   that leaks on every call today.
4. Fix `storage.Numeric` (`storage.go:22`), which admits only `float32|float64|int32|int8`
   while `DType` declares `I64`, `I16`, `Bool` and `DType.Size()` handles them — `dtypeOf`
   panics for exactly the types the enum advertises.
5. Add a `dtype` concept to `Tensor` itself; there is none today.
6. F16/BF16 bit layout defined, arithmetic deferred.

**Gate:** gradcheck passes in F64. An ill-conditioned sum is measurably more accurate in F64
than F32. F32 results are bit-identical to P5.

---

### P7 — Performance and release

1. Replace the ~20 `op_stubs.go` round-trip ops with real kernels. Each currently does
   D2H + host compute + H2D + full device sync, and leaks a pool block per call.
2. `reduction.cu` and `contiguous.cu` `cudaMalloc` per call, bypassing the pool entirely —
   route through it.
3. Pinned host memory for transfers; non-blocking stream creation; remove the
   per-op `cudaStreamSynchronize` in `add.cu:93` and `reduction.cu:75,84`.
4. Fused `CrossEntropyWithLogits`. Today `nn/loss.go:25` adds `1e-15` to a softmax output that
   float32 rounds to exactly zero, and `Log`'s backward divides by it — a ~1e15 gradient,
   invisible in the forward loss. **The backend already implements a numerically stable
   `LogSoftmax` (`cpu/ops_activation_softmax.go:61-85`) that nothing ever calls.**
5. Kaiming/He init — every model in the repo is ReLU-based, but `RandomInit`
   (`tensor.go:209`) hardcodes Glorot. For `Linear(1568,128)` that's a ~3.5× variance deficit.
   Note `fanIn`/`fanOut` are also swapped relative to `Linear`'s `[in,out]` layout — harmless
   for symmetric Xavier, a real bug the moment Kaiming lands.
6. AdamW / weight decay, momentum + Nesterov for SGD, gradient clipping, LR schedulers.
   `Optimizer` is `Step()`/`ZeroGrad()` only — no `SetLR`, so callers mutate `opt.LR` directly.
7. `internal/pools/pool.go:55` files non-power-of-two buffers into power-of-two buckets,
   degrading the pool to a ~0% hit rate.
8. Benchmarks vs. PyTorch; `README`; `CONTRIBUTING`; semver `v0.1.0`; godoc.

---

## Standing verification gates

```bash
go build ./...                                   # linux, windows, darwin
CGO_ENABLED=0 go build ./...                     # CPU-only consumers
go vet ./...
go test -race ./...
go test -run Gradcheck ./...                     # after P0
go test -tags cuda ./internal/backend/cuda/...   # on the CUDA box
go test ./example/MNIST/                         # >95% canary, after P0 fixes it
```

## Biggest risks

- **P1 changes numbers.** Land the gradcheck *first*, and change one op per commit. If a
  number moves and gradcheck still passes, the old number was wrong — that is the point.
- **P3 is the sprawling `unsafe` diff.** Its failures are silent corruption, not compile
  errors. Mitigation: P2's `Stats()` gate + `compute-sanitizer` + never changing dtype and
  ownership in the same phase.
- **Scope creep at P4.** It's the only phase that breaks API compatibility; batch every
  breaking change into it and cut a `v0.1.0` immediately after.
- **The MNIST/CIFAR canaries can't run in *fresh* CI** — `.gitignore` excludes `data/`, and the
  tests `assert.NoError` on a failed dataset load rather than skipping. Fix the skip behaviour
  in P0 (cache the dataset or skip cleanly when absent).

## Canary results — 2026-08-13 (first run since the Storage migration)

Once `example/` compiled again, both MNIST canaries were run. **Both pass, and they are far
cheaper than this document originally assumed** (it estimated "tens of minutes to hours"):

| Test | Result | Gate | Wall clock |
|---|---|---|---|
| `TestMNIST` (MLP) | **95.53%** | >95% | **10.5 s** |
| `TestMNISTCNN` | **98.70%** | >80% | **98.9 s** |

Consequences for the plan:

1. **Wire `TestMNIST` in as a per-commit regression gate now.** At 10 s it is the cheapest
   whole-stack check available and currently the only one. `TestMNISTCNN` fits a pre-merge job.
2. **The MatOperand refactor is validated by a real training run**, not just unit tests —
   `MatMulAddBias`'s backward routes both gradients through `.T()` operands.
3. **These paths are now known-good end to end:** Linear, ReLU, Softmax, cross-entropy, Adam,
   transposed matmul, broadcast-`Add` (bias), Conv2D fwd+bwd, MaxPool2D, Flatten/Reshape.
4. **Therefore every remaining P1 bug lives in API surface neither canary touches:** `Slice`,
   `BroadcastTo` views, `Reshape` on non-contiguous input, `Inverse`, `Add`'s stride-ignoring
   fast path (reachable only with a view operand), and directly-called `Broadcast*Op`.
   A gradcheck restricted to contiguous inputs would pass today and catch none of them —
   the {transposed, sliced, offset, broadcast, reshaped} input matrix *is* the deliverable.
5. Caveat: passing a canary is evidence, not proof. A gradient wrong by a constant factor can
   still train. Conv/MaxPool backward remain unverified against finite differences.

**Note:** a `-run` pattern that matches nothing exits 0. The first canary attempt reported
`PASS` having executed zero tests. Any CI gate must assert that tests actually ran.

## Suggested first commit

`engine.go:56` — skip `Backward()` when `grad == nil`. One line, fixes a live crash, and is a
clean warm-up for the gradcheck harness that has to land right after it.
