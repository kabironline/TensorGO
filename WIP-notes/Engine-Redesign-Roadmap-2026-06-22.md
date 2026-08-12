# nanograd Engine Redesign — Storage-Unified, Multi-dtype, Leak-Proof Core

> ⚠️ **SUPERSEDED 2026-08-12 by [Production-Readiness-Roadmap-2026-08-12.md](./Production-Readiness-Roadmap-2026-08-12.md).**
> A full re-audit found the engine computes silently wrong gradients in `Add`, `MatMul`,
> `Slice`, and `Inverse`; that `go build ./...` fails outright (no CUDA build tags); and that
> no finite-difference gradient check exists anywhere in the repo. Correctness and test
> infrastructure must precede the memory and dtype work this document plans.
> **The design decisions below remain valid and are carried forward** — Storage as a typed
> buffer, asymmetric CPU/GPU ownership, `Scope` for GPU lifetime, runtime dtype tags over
> generic `Tensor[T]`. Only the *sequencing* changed: this doc's P4 (Scope) becomes the new
> P2, its P2 (Storage-typed memory) becomes the new P3, and its P5 (multi-dtype) becomes P6.
> Read this one for the *why* behind the design; read the new one for what to do next.

> Roadmap authored 2026-06-22. Working mode: Claude is architect & reviewer; **you write every line**.
> Companion doc: [Tensor-Storage-Migration.md](./Tensor-Storage-Migration.md) — the concrete "how I update the Tensor" walkthrough.

## Why this redesign

Three architectural problems block nanograd from growing toward a PyTorch-like engine:

1. **GPU memory leaks** — no ownership model. Reclaiming GPU memory depends on manually calling `ClearGraph()`/`ClearComputationGraph()` every step (`tensor/tensor.go:255-317`) and every backward closure hand-`Free()`ing temporaries. `BackProp()` (`tensor/engine.go`) does *no* teardown — miss a call and it leaks. Confirmed concrete leaks: `op_stubs.go` fallback buffers never freed; `reduction.cu` `cudaMalloc`s outside the pool; `contiguous.cu` `cudaFree`s before its async kernel finishes (use-after-free); final cleanup relies on non-deterministic `runtime.SetFinalizer`.
2. **float32 welded into the type system** — `Data []float32`/`Grad []float32` (`tensor/tensor.go:10-11`), the whole `Backend` interface, every cgo call, `cublasSgemm`/`blas32.Gemm`. No `dtype` concept to branch on.
3. **Broken serialization** — `nn/linear.go` `Save` reads `Weight.Data` directly; for a CUDA tensor that's a *device pointer* → serializes garbage. No optimizer state saved (Adam `M`/`V`/`T` lost → can't resume).

**Key insight:** dtype and memory-ownership are the same problem. One new abstraction — **`Storage`**, a flat typed buffer — solves both. We steal PyTorch's *idea* (separate the buffer from the view; runtime dtype; bulk-free transient device memory) but render it in Go's grain, **not** as a C++ refcounted smart pointer. Today `Data []float32` does four jobs: names the type, counts elements via `len`, is the GC-tracked memory, and on CUDA is secretly a device pointer disguised as a slice (`ToDevice` returns `unsafe.Slice((*float32)(ptr), size)` over GPU memory). `Storage` splits those four jobs cleanly.

**The Go-grain decision that drives everything: CPU and GPU memory are asymmetric.** Go's GC already frees CPU memory correctly (your current `ClearGraph` only frees `if IsGPU()` — the CPU path *never leaked*). So CPU storage is just a GC-owned `[]byte` with zero ownership machinery; only GPU memory (off-heap, GC-invisible) needs explicit lifetime. **The whole leak problem and the entire `Scope` mechanism live on the GPU side.**

## North-Star architecture

- **`Tensor`** = non-owning **view**: `{ data *Storage; Grad *Storage; Shape, Strides, Offset; Parents; Backward; RequiresGrad }`. Shape/Strides/Offset are the *interpretation*; the bytes live in `Storage`. Many views, one buffer (zero-copy reshape/transpose/slice).
- **`Storage`** = a bag of bytes + a label. **Start dead simple (Phase 1, CPU): `{ data []byte; dtype DType; numel int }`** — the data is in `data`, one place. One `unsafe.Slice` in `F32()` reinterprets it at the kernel boundary. The GPU representation is **deferred to Phase 2**, where the single `data` field generalizes (behind a tiny buffer interface) to hold "RAM bytes **or** a GPU handle" — the answer to "where's the data" stays one field. (See migration doc.)
- **`Backend`** = runtime-polymorphic; Storage-typed signatures; each method opens with `switch a.DType()` — just an honest, greppable dispatch (Go's answer to per-type kernels), with generics inside the leaf kernels.
- **Autograd** = define-by-run unchanged; backward closures allocate transients normally. On CPU the GC reclaims them; on GPU one `defer scope.Release()` per step does. `ClearGraph` deleted.

### Resolved decisions (rejected alternative in one line)
1. **Start with the simplest `Storage` that works (`data []byte` + dtype + numel), grow the `data` field for GPU in Phase 2.** *(Rejected: shipping the full GPU-ready struct in Phase 1 — front-loads confusion for memory you won't touch until Phase 2; KISS.)*
2. **CPU bytes are a plain GC-owned `[]byte`; GPU bytes (Phase 2) hide behind the same one field via a small buffer interface — never a `[]float32` field, never a raw `unsafe.Pointer` with a keep-alive hack.** *(Rejected: a `[]float32` field re-welds float32; a bare `unsafe.Pointer` throws away the GC where it already works.)*
3. **dtype = runtime tag, `switch`-dispatched per method.** *(Rejected: `Tensor[T]` generics — interface methods can't be generic; mixed-dtype `[]*Tensor`/Module/Adam unrepresentable.)* Generics live **only inside leaf CPU kernels**.
4. **Asymmetric ownership:** CPU = let the GC do it (no refcount, no `Retain`/`Release`); GPU = `Scope` bulk-frees transients, long-lived owners hold params/optimizer state. *(Rejected: PyTorch-style refcounting everywhere — a C++ tax that solves a problem Go's GC already solves for the 90% CPU case.)*
5. **CUDA F32-only, explicit panic on non-F32 + deliberate `Cast` op.** *(Rejected: silent CPU fallback — the exact `op_stubs.go` leak pattern.)*

## Status — updated 2026-06-23

**P1, P3, and P2 (CPU) are done.** P1+P3 were a combined pass: build the simple `Storage` + `dtype` enum + `StorageFrom[T]`, route `Tensor.data`/`grad` through it, migrate every consumer (`tensor/`, `nn/`, `optim/`, tests) via the transitional `Data()`/`Grad()` shims; XOR trains end-to-end on CPU.

**P2** then introduced the Storage-typed allocation layer:
- **`storage` is now its own package** (`github.com/kabironline/nanograd/storage`), extracted from `tensor` to break the `backend <- tensor` import cycle. `tensor` keeps transitional aliases (`type Storage = storage.Storage`, forwarding `StorageFrom`) so no other tensor files changed.
- **`storage.Buffer` interface** (`Bytes()`/`Len()`/`Free()`) with a CPU `hostBuffer`; `Storage` now holds a `Buffer` so it can later hold a device handle. `Bytes()`/`F32()` panic on a (future) device buffer.
- **`backend.StorageManager`** (`AllocStorage(numel,dt)`/`FreeStorage`/`CopyStorage`) added to the core `Backend` interface; CPU + CUDA implement it. **CUDA `gpuBuffer`** (`Bytes()` panics — honest device handle, no fake slice) is **written but unverified here** (`CGO_ENABLED=0`).

⚠️ **The new Storage memory API is ADDED but NOT YET ADOPTED.** The old `[]float32` `Allocate`/`Free`/`Copy` are still the in-use path on **both** backends — and on CUDA they hand back the *fake slice over device memory*, which is exactly what the still-`[]float32` compute ops need via `.F32()`. We initially routed `tensor`'s grad/empty allocation through `AllocStorage`, but that **panics on CUDA** (the honest `gpuBuffer.Bytes()` panics while compute still calls `F32()`), so it was **reverted** — tensor allocation stays on the `StorageFrom(dev.Allocate(...))` bridge.

**Key realization:** on GPU, the memory migration is *coupled* to the compute migration. The honest `gpuBuffer` can't be adopted until the compute ops take `*Storage` and stop calling `F32()` on device memory. So `StorageManager`/`gpuBuffer` sit as the **target API**, switched on during P5+.

Still deferred:
- **P0 (allocator `Stats`)** — GPU-leak instrument; do before P4.
- **Adopting `AllocStorage`** (replacing the `[]float32` allocate/compute path) — couples to the compute-op migration (P5+).
- Added **`tensor.FromData(...)`** for external Tensor construction (the `data` field is unexported).

## Phased plan (strangler-fig — every phase ends compilable, full suite green)

| Phase | Status | Goal | Proof gate |
|---|---|---|---|
| **P0** Make leaks visible | ⬜ todo (before P4) | `Stats(){BytesInUse,BytesReserved,ActiveBlocks}` on both backends (CUDA already tracks `activeBlocks`/`currentPoolSize`) | `stats_test.go`: baseline returns to zero |
| **P1** `dtype` + `storage` packages (F32, CPU-only) | ✅ done | `DType` enum + metadata table; the simple `Storage{ data []byte; dtype; numel }`; `StorageFrom[T]`/`.F32()`/`.Bytes()`. Define F16/BF16 bit-layout, defer arithmetic. Scalars carried as `float64`. | `storage_test.go`: F32() aliasing, dtype-mismatch panic |
| **P2** Storage-typed memory methods (GPU representation arrives) | ◐ API added, **not adopted** (couples to compute migration) | `storage` extracted to own package; `storage.Buffer` + CPU `hostBuffer`; `backend.StorageManager` (CPU+CUDA); CUDA `gpuBuffer`. But tensor/compute still use the old `[]float32` path (CUDA fake slice) — adopting `AllocStorage` waits for Storage-typed compute (P5+) | `storage_test.go` + `storage_alloc_test.go` green; `.F32()`/`Bytes()` panic on a device buffer |
| **P3** Route `Tensor.Data`/`Grad` through Storage ⚠️ | ✅ done (CPU, F32) | Swap fields; transitional `Data()`/`Grad()` shims; migrate `tensor/ops_*.go` + `nn/` + `optim/` file-by-file. | `go test ./tensor/... ./optim/...` green; XOR trains end-to-end on CPU |
| **P4** GPU lifetime: the Scope | ⬜ todo | `dev.Step(func(a *Scope){...})` with internal `defer a.Release()` bulk-frees transient GPU buffers (CPU already GC'd); **delete** `ClearGraph`/`ClearComputationGraph` + `SetFinalizer`. Ride-along: fix `op_stubs.go`, `reduction.cu` (scope scratch), `contiguous.cu` (`cudaStreamSynchronize` before `cudaFree`); audit `convolution.cu`. | leak test passes on CUDA *without* `ClearGraph()` in the loop |
| **P5** Second dtype: F64 on CPU | ⬜ **next** | `switch a.DType()` → `addF32`/`addF64`; matmul splits `blas32.Gemm`/`blas64.Gemm`; `Cast` op. CUDA panics on F64. | `dtype_test.go`: F64-vs-F32 agreement + ill-conditioned sum more accurate in F64 |
| **P6** Remove bridges + Serialization v2 | ⬜ todo | Delete `[]float32` wrappers, `Data()`/`Grad()` shims, ClearGraph. Single `.safetensors`, structure in `__metadata__`; `StatefulOptimizer{StateDict/LoadStateDict}` for Adam M/V/T resume; every tensor via `Storage.ToHostBytes()`/`CopyFromHostBytes()`; atomic tmp+rename. | `serialize_test.go`: GPU tensors save real values; resumed Adam reproduces pre-save step |

## Reuse what exists
- CUDA power-of-2 caching allocator (`memory_pool.go`) — keeps; just add stats and route everything through it.
- `github.com/nlpodyssey/safetensors` (already a dep) — checkpoint container, already dtype-tagged.
- `gonum` `blas32`/`blas64` — CPU matmul per dtype.
- Define-by-run autograd (`engine.go`) — sound; keep the topological-sort backprop, only its memory teardown changes.

## Biggest risk
**Phase 3** is a sprawling `unsafe`/ownership diff whose mistakes are *silent* corruption, not compile errors. De-risk: (1) build P0's leak gate first; (2) keep the `Data()` shim through P3 so the compiler error list is your worklist; (3) never change dtype + ownership in the same phase (P3 is F32-only → any numeric break is a memory bug); (4) MNIST >95% canary (`example/MNIST/MNIST_NN_test.go`) after every phase.

## Verification (run after every phase)
```
go build ./...
go test ./...
go test ./example/MNIST/   # >95% accuracy canary
```

## What this sets up (engine-only)
Typed, owned `Storage` + explicit `Cast` + resumable optimizer state = the substrate mixed-precision (fp16/bf16 activations + fp32 master weights) and large-model checkpointing later need — without committing to any attention/transformer/tokenizer design now.
