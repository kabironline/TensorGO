# Resume Point — 2026-09-05

Written before moving to another machine. Read this first, then
[Engine-Roadmap.md](./Engine-Roadmap.md) for the full plan.

---

## ⚠️ Before you leave this machine

**There is uncommitted work.** Commit or stash it — it is not in any commit:

```
 M internal/backend/cuda/ops_memory.go           # ToCPU pointer diagnostics
 M internal/backend/cuda/test/ops_conv_test.go   # SetDefaultBackend fixes
 M internal/backend/cuda/test/ops_dmas_test.go   # SetDefaultBackend fixes
 M internal/backend/cuda/test/ops_matrix_test.go # SetDefaultBackend fixes
 M tensor/tensor.go                              # NewIdentityTensor GPU fix
 M WIP-notes/Engine-Roadmap.md                   # status update
```

Together these are **the TestCudaDMAS fix** — the long-standing
`ToCPU copy failed: 1`. Losing them means re-deriving that diagnosis. Suggested:

```bash
git add -A && git commit -m "Fix TestCudaDMAS: set default backend in CUDA tests; NewIdentityTensor on GPU"
```

Branch is `engine-rewrite`. Last commit: `af82430 Implemented inverse in cuda`.

---

## Where the project stands

| | |
|---|---|
| Phase | P0 ✅ · P1 ✅ · **P2 is next** · P3–P7 pending |
| CPU tests | 45 top-level, 0 failures · **47 gradcheck assertions, 0 skips** |
| CUDA tests | **23, all passing** (needs a GPU) |
| Coverage | `tensor` 59.3% |
| Canaries | MNIST MLP 96.7% (~10 s) · MNIST CNN 98.7% (~99 s) |
| Builds | `CGO_ENABLED=0` on Linux/Windows/macOS ✅ · `-tags cuda` ✅ |

The engine is **correct on CPU and verified by finite differences**. That was not
true at the start of this work: `MatMul`, `Add`, `Slice`, `Inverse`, `Reshape`,
and `MatVecMul` all had wrong or dropped gradients, and no test could see it.

---

## Setting up the new machine

Four things are **gitignored** and must be regenerated. Nothing builds or runs
without them.

### 1. CUDA kernels — `libcuda.a`

```bash
make kernels          # or: cd internal/backend/cuda/kernels && make all
```

**Check the GPU architecture first.** `kernels/Makefile:3` hardcodes
`-arch=sm_86` (RTX 3070, Ampere). If the new machine has a different GPU this
must change or the kernels will not run:

| GPU | Flag |
|---|---|
| RTX 20xx (Turing) | `sm_75` |
| RTX 30xx (Ampere) | `sm_86` |
| RTX 40xx (Ada) | `sm_89` |
| RTX 50xx (Blackwell) | `sm_120` |

`nvidia-smi --query-gpu=name --format=csv,noheader` tells you the card.

### 2. Datasets

Both canaries **skip cleanly** when data is absent, so a missing dataset shows up
as a skipped test rather than a failure. To actually run them:

```bash
# MNIST -> example/MNIST/data/
mkdir -p example/MNIST/data && cd example/MNIST/data
base="https://storage.googleapis.com/cvdf-datasets/mnist"
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz  t10k-labels-idx1-ubyte.gz; do
  curl -fsSL -O "$base/$f"
done
```

CIFAR-10 goes in `example/CIFAR-10/data/` (the binary distribution:
`data_batch_1..5.bin`, `test_batch.bin`).

`.github/workflows/ci.yml` has the MNIST fetch already scripted.

### 3. Toolchain quirks on the machine this was developed on

These are **environment facts, not project requirements** — re-check them:

- Work happened in **WSL** (`/mnt/e/Projects/nanograd`) for anything CUDA. The
  Windows side has `CGO_ENABLED=0`, so it can build and test CPU only.
- **`nvcc` was not on `PATH`** — it lives at `/usr/local/cuda/bin/nvcc`.
- **Go was not on `PATH`** in a non-login WSL shell — `/usr/local/go/bin`.
- So CUDA commands looked like:
  ```bash
  export PATH=$PATH:/usr/local/go/bin:/usr/local/cuda/bin
  CGO_ENABLED=1 go test -tags cuda ./internal/backend/cuda/...
  ```

### 4. Hardcoded cgo paths

Ten files under `internal/backend/cuda/` carry
`-L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib` in their `#cgo LDFLAGS`. If the new
machine puts CUDA elsewhere, override at build time rather than editing them:

```bash
make build-cuda CUDA_PATH=/opt/cuda
```

Making those paths properly discoverable is still open (P0 item 2 was only
partially done — the Makefile passes `CGO_CFLAGS`/`CGO_LDFLAGS`, but the source
still has the literals).

---

## Verifying the move worked

```bash
make                    # fmt, vet, build, test  -- CPU only, no GPU needed
make build-portable     # proves CGO_ENABLED=0 still compiles
make gradcheck          # 47 finite-difference checks, 0 skips
make canary             # MNIST MLP, expect ~96%, ~10s

# GPU (after `make kernels`):
export PATH=$PATH:/usr/local/go/bin:/usr/local/cuda/bin
CGO_ENABLED=1 go build -tags cuda ./...
CGO_ENABLED=1 go test -tags cuda -count=1 ./internal/backend/cuda/...   # 23 tests
```

**Build tags in use:** `cuda` (CUDA backend + its tests) and `examples` (the
MNIST/CIFAR canaries). Without a tag those files are excluded — which means
`go test ./...` silently runs *neither*. That is deliberate, but remember it.

---

## What was completed

### P0 — build + observability (done 2026-08-13)

The library previously **could not be built by anyone**: `go build ./...` failed
because the CUDA package had no build tags, and hardcoded Linux paths meant it
could not compile on Windows or macOS at all.

- CUDA behind `//go:build cuda`, with `cuda_stub.go` returning
  `ErrCUDAUnavailable` — `go get` now works with no CUDA toolchain
- Fixed compile breaks in all four `example/` tests and `nn/tests`
- Tests moved into `test/` subpackages, `-coverpkg` wired into Makefile + CI so
  coverage is actually attributed (it read 0.0% before)
- **`gradcheck` package** — central finite differences, position-weighted loss
- Seedable RNG (`Seed(uint64)` on `RandomOps`) so results are reproducible
- CI: 3 OSes × `CGO_ENABLED={0,1}`, vet, gofmt, race, gradcheck, coverage, canary
- README, Makefile, hygiene

### P1 — autograd correctness (done 2026-08-17)

Two structural decisions, both adopted:

- **Gradients are always logical-order, contiguous, `TotalSize(Shape)`.**
- **One offset representation** — `Slice` carries it in the storage;
  `Tensor.Offset` deleted entirely (it was provably always zero).

Fixed: `MatMul` leading dimension (via the new `MatOperand`), `MatVecMul`/
`VecMatMul` dropped gradients, `Slice` double-offset and unzeroed backward
buffer, `Inverse` backward reading the forward result, `Reshape` on a view losing
the gradient entirely, `engine.go` nil-grad crash, the `AccumulateGrad`
two-conventions bug, `Add`'s stride-ignoring fast path, `Broadcast*Op` having no
backward, plus an API tidy (`IsContiguous`, `Detach`, `Clone`, `BackPropWith`,
`BackProp` now rejects a non-scalar root).

### CUDA matrix inverse (committed, `af82430`)

Two paths with a measured crossover at n = 32:

- **small** — Gauss-Jordan on `[A|I]` entirely in shared memory, one block
- **large** — cuBLAS `getrfBatched` + `getriBatched`

```
                     n=8       n=32
small             135 µs     187 µs
large             420 µs     421 µs
```

Note the large path is flat from 8→32 — that is pure launch/setup overhead, which
is what the shared-memory kernel exists to avoid.

`cublasSgetrfBatched` only reports singularity for an **exactly** zero pivot, so a
`lu_singular_check_kernel` was added to give the LU path the same detection the
Gauss-Jordan path has.

### TestCudaDMAS (fixed, **uncommitted**)

Failed for months with `ToCPU copy failed: 1`. **Not a memory bug.** The tests
registered the CUDA backend but never made it the *default*, and
`AutoSelectBackend` prefers `"cpu"` — so the tensors were built on the host and
`ToCPU` was handed host pointers.

Fixing it exposed a real second bug: `NewIdentityTensor` wrote its diagonal
straight into device memory from Go, with the correct GPU branch sitting right
above it commented out as `// TODO: IMPLMENET LATER`.

`ToCPU` now calls `cudaPointerGetAttributes` on failure and says *which* kind of
pointer it got. That is what turned a months-old mystery into a one-line fix.

---

## What is next: P2 — ownership & lifetime

**The one question the code cannot answer: who owns a buffer?**

Four ops create a tensor that both shares the parent's storage *and* sets
`Parents` — `Transpose`, `Reshape`, `BroadcastTo`, `Slice`. And
`clearGraphHelper` treats "has parents" as "is a temporary, free its data". So a
**view of a parameter gets its buffer freed** while the parameter still needs it.
The code's own comment calls this "a better heuristic", which is an honest
admission it is guesswork.

Also open: `runtime.SetFinalizer` can `Clear()` the pool while live tensors point
into it; `MemoryPool.Clear()` only frees idle buckets so checked-out blocks leak;
Adam's `M`/`V` are never freed and there is no `Close()`; `Tensor.To` leaks the
old buffer; `ReduceSumTo` sometimes returns its input so callers free
inconsistently.

**The fix is a `Scope`** (arena/region), not refcounting — CPU memory is already
handled correctly by Go's GC, so the whole mechanism is GPU-only:

```go
dev.Step(func(s *backend.Scope) {
    loss := nn.CrossEntropyLoss(model.Forward(x), y)
    loss.BackProp()
    opt.Step()
})   // every device buffer allocated inside is released here
```

It is *positional* rather than *inferential*: "allocated inside this step" is a
fact the allocator observes; "is this an intermediate" is a guess that is
currently wrong.

### Order of work

1. **`Stats()` first** — `{BytesInUse, BytesReserved, ActiveBlocks}`. The pool
   already tracks `activeBlocks` and `currentPoolSize`, so this is ~15 lines. It
   is the gate everything else is measured against.
2. Ownership bit on `Storage`; views never free
3. `Storage.Free` idempotent (it never nils `buf` today)
4. `Scope` + `dev.Step`; delete `ClearGraph`, `ClearComputationGraph`, `SetFinalizer`
5. `Tensor.To` returns a new tensor and frees the source
6. `ReduceSumTo` gets an unambiguous ownership contract
7. Adam `Close()` + the `model.To(cuda)`-after-`NewAdam` device hazard
8. `.cu` lifetime fixes: `contiguous.cu` frees before its async kernel finishes;
   `convolution.cu`'s global workspace is freed with no stream sync and no mutex;
   `reduction.cu` `cudaMalloc`s outside the pool

**Gate:** N training steps with **no** `ClearGraph()` and `Stats()` back to
baseline; `-race` green.

⚠️ **The canaries will not catch a P2 regression.** MNIST trains fine while
leaking. `Stats()` and `-race` are the only detectors — which is why `Stats()` is
step 1.

---

## Known-open items (not blocking P2)

- **`LinAlgOps` is in the *required* `Backend` interface.** CUDA implements
  `Inverse` so the build works, but a ROCm/Metal backend would be forced to
  implement matrix inversion. Should be optional + type-asserted (P4 item 4).
- **`Contiguous(t)` is exported and used by `nn/conv`.** It is now graph-aware
  (delegates to `Clone`), so the gradient-sink hazard is gone, but the naming is
  still confusing next to `t.IsContiguous()`.
- **`cuda_inverse` allocates per call** — one `cudaMalloc`/`cudaFree` for its info
  flag, five in the LU path. At n=8 the 135 µs is almost entirely that overhead.
  Filed under P7; would be fixed naturally by P2's workspace work.
- **23 `op_stubs.go` ops round-trip to the CPU** (device→host→compute→host→device
  + full sync, leaking a pool block each). `Sigmoid`, `Tanh`, `Softmax`, and
  `MaxPool2d` backwards all run on the CPU today even on GPU. P7.

---

## Working agreement

**Claude is architect & reviewer; you write the code.** In practice this session
drifted toward Claude implementing and you reviewing — both modes worked, but the
handoffs where you wrote the code and Claude reviewed caught the most bugs
(the `AccumulateGrad` regression, the `Reshape` receiver reassignment).

## Lessons worth not relearning

Full list is at the bottom of [Engine-Roadmap.md](./Engine-Roadmap.md). The ones
that cost the most time:

1. **A symmetric loss cannot detect a permutation bug.** Gradcheck's first version
   reduced with `Sum()`, which is permutation-invariant, so a wrong-order read
   produced an identical loss. Position-weighted sum fixed it.
2. **`go test -run <pattern-that-matches-nothing>` exits 0.** A canary once
   reported `PASS` having run zero tests.
3. **A malformed build constraint silently becomes a comment.** `// go:build cuda`
   (with a space) and `//go:build cuda examples` (missing `&&`) both disabled
   their tags.
4. **An ambiguous error code can hide a trivial bug for months.**
   `ToCPU copy failed: 1` survived a `git stash` bisect and two audits.
5. **Fixing the first failure in a suite reveals the next.** `TestCudaDMAS`
   panicking aborted the package, so nine later tests had never run — one of them
   found a real segfault.
6. **A passing canary is evidence, not proof.** MNIST hit 95.5% while `Slice`,
   `Inverse`, and `Add`-on-views all had wrong gradients.
