# nanograd Engine Roadmap

> **Living document.** Merges the June 2026 engine-redesign plan (design rationale)
> with the August 2026 production-readiness plan (sequencing and audit findings).
> Supersedes `Engine-Redesign-Roadmap-2026-06-22.md` and
> `Production-Readiness-Roadmap-2026-08-12.md`, both removed — see git history.
>
> Working mode: Claude is architect & reviewer; **you write the code.**
> Companion: [Tensor-Storage-Migration.md](./Tensor-Storage-Migration.md).

**Status as of 2026-08-16**

| | |
|---|---|
| Builds | `CGO_ENABLED=0` on Linux/Windows/macOS ✅ · `-tags cuda` ✅ |
| Gradchecks | **35 pass, 1 skip** |
| Coverage | `tensor` 59.3% (was unmeasurable) |
| Canaries | MNIST MLP **95.5%** (10s) · MNIST CNN **98.7%** (99s) |
| Phase | P0 ✅ done · **P1 ~80%** · P2–P7 pending |

---

## Why this redesign exists

Three architectural problems block nanograd from growing into a real engine.

**1. No ownership model for GPU memory.** Reclaiming device memory depends on
manually calling `ClearGraph()`/`ClearComputationGraph()` every step and on every
backward closure hand-`Free()`ing its temporaries. `BackProp()` does no teardown —
miss a call and it leaks. Confirmed leaks: `op_stubs.go` fallback buffers, `reduction.cu`
`cudaMalloc`s outside the pool, `contiguous.cu` freeing before its async kernel
finishes (use-after-free), and final cleanup relying on non-deterministic
`runtime.SetFinalizer`.

**2. float32 welded into the type system.** `[]float32` runs through the whole
`Backend` interface, every cgo call, `cublasSgemm`/`blas32.Gemm`. There is no dtype
concept to branch on.

**3. Broken serialization.** `Save` reads tensor data directly; for a CUDA tensor
that is a *device pointer*, so it serializes garbage. No optimizer state is saved,
so training cannot resume.

**Key insight:** dtype and memory-ownership are the same problem. One abstraction —
**`Storage`**, a flat typed buffer — solves both. Today `Data []float32` does four
jobs: names the type, counts elements, is the GC-tracked memory, and on CUDA is
secretly a device pointer disguised as a slice. `Storage` splits those cleanly.

**The Go-grain decision that drives everything: CPU and GPU memory are asymmetric.**
Go's GC already frees CPU memory correctly — the CPU path never leaked. So CPU
storage is a GC-owned `[]byte` with zero ownership machinery, and **the entire leak
problem and the whole `Scope` mechanism live on the GPU side.**

### What the August audit added

The June plan assumed the engine was *correct* and needed a better memory model.
A full re-audit found otherwise: **the engine computed wrong gradients, on CPU, in
its most-used operations, and no test could see it.** That reordered everything.
Correctness and a harness that can *prove* correctness come first; multi-dtype —
originally phase 1 — is now near the end, which is the right place for it.

---

## North-star architecture

- **`Tensor`** = non-owning **view**: `{ data *Storage; grad *Storage; Shape, Strides, Offset; Parents; Backward; RequiresGrad }`. Shape/strides are the *interpretation*; bytes live in `Storage`. Many views, one buffer.
- **`Storage`** = bytes + a dtype label, behind a small `Buffer` interface: CPU returns its slice, CUDA holds a device handle and panics on `Bytes()`.
- **`Backend`** = runtime-polymorphic, Storage-typed, each method opening with `switch a.DType()` — an honest, greppable dispatch, with generics inside leaf kernels.
- **Autograd** = define-by-run, unchanged. Backward closures allocate transients normally; on CPU the GC reclaims them, on GPU one `defer scope.Release()` per step does.

### Resolved decisions (rejected alternative in one line)

1. **Simplest `Storage` that works** (`[]byte` + dtype + numel), growing the buffer field for GPU. *(Rejected: shipping the full GPU-ready struct up front — front-loads confusion.)*
2. **CPU bytes are a GC-owned `[]byte`; GPU bytes hide behind the same field via a `Buffer` interface.** *(Rejected: a `[]float32` field re-welds float32; a bare `unsafe.Pointer` throws away the GC where it already works.)*
3. **dtype = runtime tag, `switch`-dispatched.** *(Rejected: `Tensor[T]` generics — interface methods can't be generic, and a mixed-dtype `[]*Tensor`/Module/Adam is unrepresentable.)*
4. **Asymmetric ownership:** CPU = let the GC do it; GPU = `Scope` bulk-frees transients. *(Rejected: refcounting everywhere — a C++ tax solving a problem Go's GC already solves.)*
5. **CUDA F32-only, explicit panic on non-F32, deliberate `Cast`.** *(Rejected: silent CPU fallback — the exact `op_stubs.go` pattern that leaks per call.)*
6. **Descriptors only where the kernel can exploit them.** `MatMul` takes `MatOperand` (gemm has native `lda`+trans); `Inverse` takes a plain slice (LU needs packed scratch anyway). *(Rejected: a uniform descriptor everywhere — a struct field you accept and silently ignore is worse than a parameter that was never there.)*
7. **Backward formulas are tensor expressions, not device kernels.** A backward needs its own kernel only for data-dependent indexing (maxpool argmax, gather/scatter) or where composition is unacceptably slow (conv). *(Rejected: one backward kernel per op per backend — the reason half the CUDA backwards are host round-trips today.)*

---

## The phased plan

Every phase ends compilable on all three OSes with the suite green.

| Phase | Theme | Status |
|---|---|---|
| **P0** | Build, test infrastructure, CI | ✅ **done** |
| **P1** | Autograd correctness (CPU) | ◐ **~80%** |
| **P2** | Ownership & lifetime (the `Scope`) | ⬜ |
| **P3** | Device safety — kill the fake slice | ⬜ |
| **P4** | Errors & API surface (only breaking phase) | ⬜ |
| **P5** | Serialization v2 | ⬜ |
| **P6** | Multi-dtype | ⬜ |
| **P7** | Performance & release | ⬜ |

---

### P0 — Build and observability ✅ done 2026-08-13

1. **CUDA behind `//go:build cuda`** with a `cuda_stub.go` exposing `NewCUDABackend`, `GetCudaDeviceCount`, `GetCudaDeviceProps` and `ErrCUDAUnavailable`. `go get` now works with no CUDA toolchain.
2. **`CUDA_PATH`** plumbed via `CGO_CFLAGS`/`CGO_LDFLAGS` in the Makefile.
3. **Compile breaks fixed** — all four `example/` tests and `nn/tests`.
4. **Tests in `test/` subpackages** matching `internal/backend/cuda/test`, with `-coverpkg` wired into Makefile and CI so coverage is still attributed.
5. **`gradcheck` package** — central finite differences, position-weighted loss.
6. **Seedable RNG** — `Seed(uint64)` on `RandomOps`; the CPU backend owns a mutex-guarded source.
7. **CI** — 3 OSes × `CGO_ENABLED={0,1}`, vet, gofmt, race, gradcheck, coverage, MNIST canary.
8. **Hygiene** — README, Makefile, empty `dtype/` removed, kernel archive gitignored, canaries skip cleanly when the dataset is absent.

Gradcheck found **three** predicted bugs on its first run. All three are now fixed.

---

### P1 — Autograd correctness ◐ in progress

Two structural decisions drive it, both now **adopted**:

- **Gradients are always logical-order, contiguous, `TotalSize(Shape)`.** `ensureGrad` and `AllocGrad` merged; `Transpose`'s backward permutes into logical order rather than scattering physically.
- **One offset representation.** `Slice` carries its offset in the storage (`Offset` stays 0) instead of doing both.

**Done**

| Fix | Was |
|---|---|
| `MatMul` leading dimension | `Strides[0]` passed as `lda`; panicked on CPU, corrupted silently on CUDA. Replaced by `MatOperand` + `asMatOperand`, collapsing `MatrixOps` to a single `MatMul(a, b, out, alpha, beta)` |
| `MatVecMul`/`VecMatMul` | Hand-built views with no `Parents`/`Backward` — the vector never trained. Now fused single-gemm ops with real backwards |
| `Slice` double offset | Resliced storage *and* set `Offset`; every read landed at `base[2*offset]` |
| `Slice` backward | Accumulated into an unzeroed pooled buffer |
| `Inverse` backward | Read the forward result instead of `out.Grad()`. Now composite: `-(Yᵀ @ gradOut @ Yᵀ)`, with `alpha = -1` folding in the negation |
| `Reshape` on a view | Reassigned the receiver to a graph-detached `Contiguous` copy — total gradient loss |
| `engine.go` nil-grad | Called `Backward()` on nodes with no gradient → nil-slice index |
| Grad layout | `ensureGrad` sized by storage, `AllocGrad` by shape; `AccumulateGrad` stored logically on one path and physically on another |

**Remaining**

1. **`Add` fast path ignores strides** — `ops_matrix.go`, the one open gradcheck skip. Guard the fast path on contiguity so a strided operand falls through to `BroadcastAddOp`, which already materialises.
2. **`AccumulateGrad` dead paths** — with the new invariant, path 3 is unreachable and path 2 always fires. Collapse to a single `Device.Add`. Pure deletion, no behaviour change; also removes the `t.grad` reassignment that orphans `ToGradTensor` handles.
3. **`Slice` backward indexes physically** — passes only because its parents are contiguous. Align to the logical rule; add slice-of-view to the gradcheck matrix.
4. **`contiguous` is a stored bool** that `Transpose`/`BroadcastTo`/`Slice` forget to set. Derive it from strides.
5. **`Contiguous(t)` detaches from the graph** — exported, so `tensor.Contiguous(x)` in user code is a silent gradient sink. It has bitten three times. Unexport it or give it a copy-backward.
6. **`BroadcastAddOp`/`Sub`/`Mul`/`DivOp`** set `Parents` but no `Backward`. Unexport or complete them.
7. **API tidy** — dedupe `ComputeStrides`/`computeStrides`; resolve `Contiguous` meaning both a package function and a method; delete the now-dead `Tensor.Offset`; add `Detach`, `Clone`; make `BackProp` reject a non-scalar root.

**Gate:** gradcheck green for every op × {contiguous, transposed, sliced, offset, broadcast, reshaped}; a test that calls `BackProp` twice; a test mixing `RequiresGrad=false` nodes.

---

### P2 — Ownership and lifetime

`Storage` has no owner. `Transpose`, `Reshape`, `BroadcastTo`, and `Slice` all set
`data: t.data` **and** `Parents: []*Tensor{t}`, and `clearGraphHelper` treats "has
Parents" as "intermediate, free its data" — so a view of a *parameter* gets its
buffer freed, while the same function's `else` branch promises not to. Two views of
one buffer make it a genuine double-free.

1. **Allocator `Stats()`** (`BytesInUse`/`BytesReserved`/`ActiveBlocks`) on both backends — build this first; it is the gate for everything else here.
2. Explicit ownership on `Storage` (owner vs. view). No refcounting on CPU.
3. `Storage.Free` idempotent — it never nils `buf`.
4. **The `Scope`**: `dev.Step(func(s *Scope){...})` + `defer s.Release()`. Delete `ClearGraph`, `ClearComputationGraph`, and the `SetFinalizer` teardown.
5. `Tensor.To` must return a new tensor and free the source — it currently mutates the receiver, leaks the old buffer, and desynchronises every existing view.
6. Unambiguous ownership contract for `ReduceSumTo` (it sometimes returns its input).
7. `Contiguous()` copies are never freed; `ReLU`/`Log`/`Square` additionally capture them in their closures, pinning a duplicate of every activation.
8. Adam's `M`/`V` are never freed; no `Close()`. Fix the device-ordering hazard where `model.To(cuda)` after `NewAdam` feeds host buffers to `cuda_step_adam`.

**Gate:** N training steps with **no** `ClearGraph()` and `Stats()` back to baseline; `-race` green.

---

### P3 — Device safety: kill the fake slice

`Allocate` returns `unsafe.Slice((*float32)(devPtr), n)` — a device pointer typed as
a host slice. Seven unguarded sites index it and would segfault on GPU. This is the
phase where `StorageManager`/`gpuBuffer` finally get **adopted**, and — exactly as the
June plan concluded — that is coupled to the compute migration.

1. Convert `Backend` op signatures from `[]float32` to `*storage.Storage`, family by family. Delete `MemoryManager`'s `[]float32` methods as each lands.
2. Delete the `Data()`/`Grad()` shims — they have **write** dependencies (`optim` writes parameters through them), so they need `CopyFrom`/`Fill` replacements, not just removal.
3. Fix `cuda_contiguous`: swapped `total`/`offset` args, and `[]int`→`*C.int` truncation (device sees `shape=[2,0]`, kernel divides by zero).
4. `runtime.LockOSThread` around device work — `cudaSetDevice` is thread-local and Go migrates goroutines. Prime suspect for the intermittent `ToCPU copy failed: 1`.
5. **Register the CUDA backend** — `NewCUDABackend` is never passed to `RegisterBackend`, so `backends["cuda"]` can never hit. Also reconcile `"cuda:0"` vs `"cuda"`.
6. Fix the global cuDNN workspace: `cudaFree` + realloc with no stream sync while a conv may still be reading it — the clearest true async use-after-free in the tree.
7. Sync policy: five unary ops bypass the stream and `cudaDeviceSynchronize` per launch; `Copy` is blocking; `ToDevice`/`WriteToDevice` issue async H2D from Go heap memory.
8. `WorkerPool`'s single shared `WaitGroup` — concurrent `Process` panics.
9. `cpu/ops_dmas.go` iterates `len(out)` and ignores the `size` argument CUDA honours.

**Gate:** `compute-sanitizer` clean; host code touching a device buffer **panics** rather than segfaults; `TestCudaDMAS` passes.

---

### P4 — Errors and API surface

The last phase that breaks compatibility. Batch every breaking change here and cut
`v0.1.0` immediately after.

1. **Draw the panic/error line.** Panic only for programmer error. Return `error` for shape/device/dtype mismatch, singular matrix, OOM, no-backend-available. (`Inverse` already does; it is the template.)
2. Device and dtype mismatch are **never checked** — `ops_matrix.go` uses `a.Device` to operate on `b.Data()`.
3. **Mutex or eliminate the backend registry** — an unsynchronised package-level map written by `RegisterBackend` and read by `AutoSelectBackend` on every tensor construction.
4. **Split the `Backend` interface.** 62 methods across 15 required sub-interfaces, while its own doc comment claims optional composition — which makes `SupportsConvolutions` and friends dead code that always returns true. Required core: memory + elementwise + matmul. Everything else optional with a real composite fallback. **`LinAlgOps` is the first instance and is currently mis-filed in the required set** — a ROCm backend cannot exist until someone writes matrix inversion for it.
5. **Make `Module` composable** — `Sequential` doesn't implement it, so it can't nest. Add `var _ Module = (*X)(nil)` assertions (their absence is why this and the dropped-conv-layer bug went unnoticed). Add `Train()`/`Eval()` before Dropout/BatchNorm exist.
6. Encapsulate `Tensor`'s exported mutable fields — callers reassign `out.Parents` after construction as normal practice.
7. Missing constructors: `Zeros`, `Ones`, `Full`, `Rand`, `Arange`, `String`.
8. `context.Context` on training-loop entry points.

---

### P5 — Serialization v2

1. `RequiresGrad` must survive a round trip — one line; today loaded models silently do not train.
2. `LoadModuleAt` must handle Conv2D/MaxPool2D/Flatten and **error** on unknown keys rather than returning `(nil, nil)` and truncating the model.
3. Serialize layer hyperparameters — `Conv2D.Stride`/`Padding` and all of `MaxPool2D`'s config are written nowhere.
4. Replace the activation-as-dummy-tensor marker with real config in `__metadata__`.
5. Populate `__metadata__`: format version, framework tag, dtype/layout contract, architecture. Reject unknown versions.
6. Hierarchical keys (`block.0.conv.weight`) to match P4's nestable modules.
7. All tensors staged through `Storage.ToHostBytes()`/`CopyFromHostBytes()` — never a raw `Data()` range, which is what breaks GPU save.
8. `StatefulOptimizer{StateDict/LoadStateDict}` for Adam `M`/`V`/`T`, LR, betas, step count, RNG state.
9. Atomic write: temp file + `fsync` + `rename`. Today a crash mid-write destroys the previous good checkpoint too.
10. Stream the write — currently ~3× model size peak RSS.

**Gate:** save→load→train reproduces the pre-save loss curve; a CNN round-trips intact; Adam resumed at step N reproduces step N+1.

---

### P6 — Multi-dtype

The original phase 1, landing last because it is only provable once gradcheck exists.

1. `switch a.DType()` dispatch per backend method; generics inside leaf CPU kernels only.
2. F64 on CPU first: `blas32.Gemm`/`blas64.Gemm` split. CUDA panics on F64.
3. Explicit `Cast` op. Never a silent CPU fallback.
4. Fix `storage.Numeric`, which admits only `float32|float64|int32|int8` while `DType` declares `I64`, `I16`, `Bool` — `dtypeOf` panics for exactly the types the enum advertises.
5. Add a dtype to `Tensor` itself; there is none today.
6. F16/BF16 bit layout defined, arithmetic deferred.

**Gate:** gradcheck passes in F64; an ill-conditioned sum is measurably more accurate in F64; F32 results bit-identical to P5.

---

### P7 — Performance and release

1. Replace the ~23 `op_stubs.go` round-trip ops with real kernels. Each does D2H + host compute + H2D + full sync **and leaks a pool block per call**. On GPU this is where `Sigmoid`, `Tanh`, `Softmax`, and `MaxPool2d` backwards actually run today.
2. `reduction.cu` and `contiguous.cu` `cudaMalloc` per call, bypassing the pool.
3. Pinned host memory; non-blocking stream; remove per-op `cudaStreamSynchronize`.
4. **Fused `CrossEntropyWithLogits`.** `nn/loss.go` adds `1e-15` to a softmax output that float32 rounds to exactly zero, and `Log`'s backward divides by it — a ~1e15 gradient, invisible in the forward loss. The backend already has a stable `LogSoftmax` that **nothing ever calls**.
5. Kaiming/He init — every model here is ReLU-based but `RandomInit` hardcodes Glorot (~3.5× variance deficit for `Linear(1568,128)`). Note `fanIn`/`fanOut` are swapped relative to `Linear`'s `[in,out]` layout — harmless for symmetric Xavier, a real bug the moment Kaiming lands.
6. AdamW/weight decay, momentum + Nesterov, gradient clipping, LR schedulers.
7. `internal/pools/pool.go` files non-power-of-two buffers into power-of-two buckets, degrading to a ~0% hit rate.
8. Benchmarks vs. PyTorch; `CONTRIBUTING`; semver `v0.1.0`; godoc.

---

## Standing verification

```bash
make                    # fmt, vet, build, test
make build-portable     # CGO_ENABLED=0 -- the one that matters before release
make gradcheck          # finite differences over every backward
make canary             # MNIST MLP, >95%, ~10s
make test-race
make cover
go build -tags cuda ./...
go test -tags cuda ./internal/backend/cuda/...   # on the GPU box
```

## Biggest risks

- **P1 changes numbers.** One op per commit; gradcheck first. If a number moves and gradcheck still passes, the old number was wrong — that is the point.
- **P3 is the sprawling `unsafe` diff.** Failures are silent corruption, not compile errors. Mitigation: P2's `Stats()` gate, `compute-sanitizer`, and never changing dtype and ownership in the same phase.
- **Scope creep at P4.** It is the only compatibility break; batch everything into it.

---

## Lessons worth not relearning

Each of these cost real time in August 2026.

1. **A symmetric loss cannot detect a permutation bug.** Gradcheck's first version reduced with `Sum()`, which is permutation-invariant — `d(sum)/dx` is 1 for every element regardless of the order an op emits them in. `Add`-on-a-transposed-view passed. Switching to a position-weighted sum (`sum(out * w)`, `w` distinct per position) surfaced it immediately.
2. **`go test -run <pattern-that-matches-nothing>` exits 0.** The first canary run reported `PASS` having executed zero tests. Any CI gate must assert tests actually ran.
3. **A malformed build constraint silently becomes a comment.** `// go:build cuda` (with a space) and `//go:build cuda examples` (missing `&&`) both disabled their tags — one silently, one as a vet error.
4. **`Contiguous(t)` returns a graph-detached tensor.** It has caused a silent gradient sink three separate times. Exported, so user code hits it too.
5. **Never accept a descriptor field you don't honour.** `Inverse` took a `MatOperand` and ignored `LD` — the same bug class as the `lda` bug it was written after.
6. **A passing canary is evidence, not proof.** MNIST hit 95.5% while `Slice`, `Inverse`, and `Add`-on-views all had wrong gradients — an MLP simply never touches those paths. Adam also normalises by gradient magnitude, so it is especially forgiving of scale errors.
