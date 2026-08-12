# How the `Tensor` Evolves: `[]float32` → `Storage`

> Companion to [Engine-Redesign-Roadmap-2026-06-22.md](./Engine-Redesign-Roadmap-2026-06-22.md).
> This is the concrete "how I update the Tensor" walkthrough. You write the code; this is the design + a fully-worked reference example you replicate across the ~40 ops.

> **STATUS (2026-06-23): this migration is DONE on CPU (F32).** `Tensor.data`/`grad` are now `*Storage`; `tensor/`, `nn/`, and `optim/` are fully migrated via the `Data()`/`Grad()` shims; all CPU tests pass and XOR trains end-to-end. The GPU representation (`DevicePtr`/Phase 2) and the `Scope` (Phase 4) are **not** built yet — the `Storage` shipped is the simple `{ data []byte; dtype; numel }` and the backend is still `[]float32`-typed (bridged by the shims). The sections below describe the full target design; treat the GPU/Scope parts as forward-looking.

---

## 1. The problem in one sentence

`Data []float32` is doing **four jobs at once**, and that conflation is the root of every leak and the float32 lock-in:

| Job | How it does it today | Why it's a problem |
|---|---|---|
| Names the element type | the `float32` in the slice type | hardcodes dtype into the type system |
| Counts elements | `len(t.Data)` | breaks the moment 1 element ≠ 4 bytes (F64, F16) |
| Is the memory | the slice's backing array | GC-managed on CPU... |
| Is a device pointer | on CUDA, `ToDevice` returns `unsafe.Slice((*float32)(devptr), n)` | ...but on GPU it's a *fake slice over `cudaMalloc` memory* the GC can't manage → the leak |

`Storage` gives each job its own field. After the migration, `[]float32` means only "a real CPU array I can index" again.

---

## 2. Target types

```go
// package storage
//
// Storage = a bag of bytes + a label saying what type they are.
// Phase 1 is CPU-only, so the bytes are just a []byte. That is the WHOLE struct.
type Storage struct {
    data  []byte      // THE DATA. it's right here — one place, always.
    dtype dtype.DType // how to read the bytes (F32, F64, ...)
    numel int         // how many elements.  len(data) == numel * dtype.Size()
}

// F32 reinterprets the byte bag as []float32 — the ONE place the concrete type is
// reintroduced, and the ONLY use of unsafe in normal code. Panics on dtype mismatch.
// This assert is your manual stand-in for the compile-time type-checking we gave up.
func (s *Storage) F32() []float32 {
    s.assert(dtype.F32)
    return unsafe.Slice((*float32)(unsafe.Pointer(&s.data[0])), s.numel)
}
func (s *Storage) Bytes() []byte      { return s.data }
func (s *Storage) DType() dtype.DType { return s.dtype }
func (s *Storage) Numel() int         { return s.numel }
```

```go
// package tensor  — Tensor becomes a non-owning VIEW
type Tensor struct {
    data    *storage.Storage // was: Data []float32
    Grad    *storage.Storage // was: Grad []float32
    Shape   []int
    Strides []int
    Offset  int
    Parents []*Tensor
    Backward func()
    Device  backend.Backend
    RequiresGrad bool
    contiguous   bool
}
```

> **"Where is the data?" → `s.data`. Always. One answer.** Don't over-think it: Storage is a byte bag (`data`) plus a label (`dtype`+`numel`). That's the entire idea, and it's all you need for Phases 0–5 (CPU).
>
> **GPU is a Phase-2 concern — deliberately not in this struct yet.** The one honest constraint: GPU memory lives *on the card*, not in your program's RAM, so a Go `[]byte` cannot point at it (pretending it can is exactly today's leak — see the `device pointer` row above). When you wire CUDA through Storage in Phase 2, the `data` field is the *only* thing that has to learn about devices — you'll generalize "a `[]byte` in RAM" to "a `[]byte` in RAM **or** a handle to bytes on the GPU" behind a tiny interface, so the answer stays "the data is in `data`." Decide the exact shape *then*, with CUDA in front of you — not now. See [§3 Phase 2](#phase-2--device-aware-allocation-gpu-representation-arrives).
>
> **Why GPU gets special treatment when it arrives:** Go's GC already frees CPU memory the moment the last Tensor referencing a `*Storage` is unreachable — the CPU path *never leaked* (proof: your current `ClearGraph` only frees `if t.Device.IsGPU()`). GPU memory is off-heap and invisible to the GC, so it needs explicit lifetime (the `Scope`, Phase 4). **The entire leak problem lives on the GPU side** — but you fix the CPU core first without ever touching it.

---

## 3. The migration is staged — you do NOT rewrite Tensor in one shot

The whole point of the strangler-fig approach is that the **compiler is your worklist**. Here's the exact sequence of what the Tensor looks like at each phase.

### Phase 1 — `Storage` exists, `Tensor` untouched
You build `storage` + `dtype` packages in isolation. `Tensor` still has `Data []float32`. Zero blast radius. Write `storage_test.go` to *feel* the unsafe/ownership rules before they matter.

### Phase 2 — device-aware allocation (GPU representation arrives)
The backend starts allocating Storage: `dev.Allocate(numel, dt) *Storage`. This is the **first** moment GPU memory has to be representable, so it's exactly where the `data` field grows up — and not one phase sooner.

- **CPU** stays trivial: `Storage{ data: make([]byte, numel*dt.Size()), ... }`.
- **CUDA** can't use `[]byte` (the bytes are on the card). So generalize the one field behind a tiny interface — this is the *whole* GPU complication, isolated to the `storage` package:
  ```go
  type buffer interface {
      Bytes() []byte  // CPU: the slice.  GPU: panics — you must copy to host first.
      Free()          // CPU: no-op (GC).  GPU: return memory to the pool (wired up in Phase 4).
  }
  // the struct's `data []byte` becomes `buf buffer`; F32() reads buf.Bytes() and still
  // works unchanged on CPU, and is simply not callable on GPU. "Where's the data?" → buf.
  ```
- Keep the old `dev.Allocate(size) []float32` as a thin CPU shim so existing code compiles:
  ```go
  func (b *CPUBackend) Allocate(size int) []float32 { return b.AllocateS(size, dtype.F32).F32() }
  ```

`Tensor` is still untouched; everything is still F32. The answer to "where is the data" is still one field — it just went from `[]byte` to "a buffer that knows where it lives."

### Phase 3 — swap the fields, add the shim, migrate op-by-op  ⚠️ the big one
1. Change `Data []float32` → `data *storage.Storage`, `Grad []float32` → `Grad *storage.Storage`.
2. Add a **transitional aliasing shim** so existing code keeps compiling:
   ```go
   // DELETE this in Phase 6. It only exists to make the migration incremental.
   func (t *Tensor) Data() []float32 { return t.data.F32() }
   ```
   Now `t.Data` (field access) becomes `t.Data()` (method call) everywhere — `go build ./...` hands you the **exact list** of call sites to fix. Migrate `tensor/ops_*.go` **one file at a time**, each ending green.
3. Everything is **still F32** in Phase 3. That's deliberate: if a test breaks, it's a *memory* bug, never a dtype bug.

> **The GPU `Module.Save()` bug fixes itself here.** `nn/linear.go` stops reading a raw device slice; it goes through `t.data` which knows it's on CUDA and copies host-side first (Phase 6 wires the actual `ToHostBytes`).

### Phase 4 — GPU lifetime goes live; delete `ClearGraph`
The GPU buffer's `scope` wiring activates (CPU was already handled by the GC the whole time). See the worked example below for what changes in a backward closure.

### Phase 5 — dispatch on dtype; add F64
The `switch a.DType()` shells go into the backend ops. `Tensor` itself barely changes — it already carries dtype via `data.DType()`.

---

## 4. Worked example: `Add` through every phase

This is the reference pattern. Replicate its shape across the other ops.

### Today (`tensor/ops_matrix.go:9`)
```go
func (a *Tensor) Add(b *Tensor) *Tensor {
    if sameShape(a.Shape, b.Shape) {
        outData := a.Device.Allocate(len(a.Data))           // 1) len = element count (breaks for F64)
        a.Device.Add(a.Data, b.Data, outData, len(a.Data))  // 2) []float32 everywhere

        out := &Tensor{ Data: outData, /* ... */ Parents: []*Tensor{a, b} }
        out.Backward = func() {
            a.AccumulateGrad(out.Grad)
            b.AccumulateGrad(out.Grad)
        }
        return out
    }
    // broadcasting path: note out.Backward calls a.Device.Free(gradA) by hand — the manual-free footgun
}
```

### Phase 3 (Storage-backed, still F32, via the shim)
```go
func (a *Tensor) Add(b *Tensor) *Tensor {
    if sameShape(a.Shape, b.Shape) {
        out := a.data.NewLike()                  // Storage of same numel+dtype+device as a.data
        a.Device.AddS(a.data, b.data, out)       // NEW Storage-typed op (dtype carried by Storage)

        t := &Tensor{ data: out, /* ... */ Parents: []*Tensor{a, b} }
        t.Backward = func() {
            a.AccumulateGrad(t.Grad)             // AccumulateGrad now takes *Storage (see §5)
            b.AccumulateGrad(t.Grad)
        }
        return t
    }
}
```
What changed: no more `len(a.Data)` for sizing (Storage knows its `numel`), no raw `[]float32`, output sizing via `NewLike()`. Numerics identical — pure plumbing.

### Phase 4 (scope owns the transients; no manual Free)
The broadcasting path's `a.Device.Free(gradA)` **disappears**. Backward allocates the reduction temp from the scope:
```go
t.Backward = func() {
    if sameShape(t.Shape, a.Shape) {
        a.AccumulateGrad(t.Grad)
    } else {
        gradA := scope.ReduceSumTo(a.data, t.Grad, t.Shape, a.Shape) // scope-owned; NEVER freed here
        a.AccumulateGrad(gradA)
        // no Free — `defer scope.Release()` in dev.Step reclaims it, even on panic
    }
}
```
The `out` Storage from the forward pass is also scope-owned (allocated inside `dev.Step`). Forget nothing, leak nothing.

> **On CPU this code path leaks nothing even without the scope** — `gradA` and `out` are GC-owned `[]byte` buffers that vanish when unreachable. The `Scope` is doing real work only when these Storages live on the GPU. Same code, but the safety net is the GC on CPU and the `Scope` on GPU.

### Phase 5 (the kernel dispatches on dtype — but this is in the *backend*, not Tensor)
`Add` in `tensor/` doesn't change. The backend does:
```go
func (b *CPUBackend) AddS(x, y, out *storage.Storage) {
    switch x.DType() {                 // the dtype dispatch switch — honest and greppable
    case dtype.F32: addKernel(x.F32(), y.F32(), out.F32())   // generics OK inside the kernel
    case dtype.F64: addKernel(x.F64(), y.F64(), out.F64())
    default: panic("AddS: unsupported dtype " + x.DType().String())
    }
}
func addKernel[T ~float32 | ~float64](x, y, out []T) { for i := range x { out[i] = x[i] + y[i] } }
```

---

## 5. The two Tensor methods that need real thought

### `AccumulateGrad` (today `tensor.go:176`)
Signature changes `grad []float32` → `grad *storage.Storage`. The three paths survive but reframed:
- **Fast path** (contiguous, offset 0): `t.Device.AddS(t.Grad, grad, t.Grad)`.
- **View slow path** (today copies to CPU via `ToCPU` and re-`ToDevice`s — itself a leak source): in Phase 4 the CPU scratch comes from the scope, not bare `ToCPU`/`ToDevice`. Long-term, replace with a backend-native scatter so views never round-trip.
- **`ensureGrad`** (today `tensor.go:320`): `t.Grad = t.Device.Allocate(len(t.Data))` becomes `t.Grad = t.data.NewLikeZeroed()`. The CPU case needs no thought — it's a GC-owned `[]byte`. The distinction only bites on GPU: a **parameter's** grad is long-lived (allocated outside any scope, zeroed by `ZeroGrad`, never released), while an **intermediate's** grad is scope-owned. *On GPU, ownership is decided at creation by who allocates it* — which is what replaces the fragile leaf-vs-intermediate heuristic in `clearGraphHelper` (`tensor.go:282`).

### `ClearGraph` / `ClearComputationGraph` / `Free` → **deleted in Phase 4**
All three (`tensor.go:244`, `:258`, `:276`) exist only to free **GPU** memory by hand — they already no-op on CPU (`if t.Device.IsGPU()`). Once `dev.Step(func(a *Scope){...})` does `defer a.Release()`, there is nothing to manually clear on either device. Your training loop goes from:
```go
optimizer.ZeroGrad(); pred := model.Forward(x); loss := nn.MSELoss(pred, y)
loss.BackProp(); loss.ClearComputationGraph(); optimizer.Step()   // forget the Clear → leak
```
to:
```go
dev.Step(func(a *Scope) {
    optimizer.ZeroGrad()
    loss := nn.MSELoss(model.Forward(x), y)
    loss.BackProp()
    optimizer.Step()
})   // transients freed here, even on panic — nothing to forget
```

---

## 6. Gotchas (the parts that bite)

1. **CPU storage is just a GC-owned `[]byte` — no unsafe bookkeeping.** Because `data` is a real Go slice, the GC keeps it alive as long as any `*Storage` references it. No `unsafe.Pointer` to keep reachable, no keep-alive field. The *only* `unsafe` is inside `F32()`/`F64()`, reinterpreting `data` at the kernel boundary, and the slice it returns still pins `data` for its lifetime. GPU (Phase 2+) is the opposite: the device pointer is off-heap, the GC can't see it, so frees MUST be explicit — which is the whole reason `Scope` exists.
2. **`.F32()` panics on GPU — on purpose.** Once the GPU buffer exists (Phase 2), `F32()`/`Bytes()` on a device Storage panic, forcing GPU code through device ops instead of pretending device memory is a slice. If you hit that panic, you found a path secretly doing the old fake-slice trick.
3. **Views share one `*Storage` — and on CPU the GC handles it for free.** `Reshape`/`Transpose`/slice must NOT allocate; they make a new `Tensor` pointing at the same `data` with different `Shape`/`Strides`/`Offset`. On CPU, the shared buffer is freed by the GC when the last view is unreachable — **no `Retain`/`Release`, no refcount** (the C++ idiom we deliberately dropped). On GPU, all views of one buffer live and die together within a step: the owning `Scope` frees it at step end, after every view is done. The only thing to police is a GPU view *escaping* its step — treat that as a bug, not a refcount problem.
4. **`numel` vs bytes.** Every `*4` and every `len()` that meant "element count" must become `s.Numel()`; every byte calculation becomes `s.Numel() * s.DType().Size()`. Grep for `* 4` and `len(` in the ops files.
5. **CUDA `contiguous.cu` race (fix in Phase 4).** The kernel launch is async; `cudaGetLastError` only checks *launch* status, not completion. Today it `cudaFree`s `d_shape`/`d_strides` while the kernel may still read them. Fix: `cudaStreamSynchronize(stream)` before the frees (or allocate them from the scope in Go and free after `Sync()`). Audit `convolution.cu` for the same shape.

---

## 7. Your Phase-3 checklist for `tensor.go` specifically

- [ ] `Data []float32` → `data *storage.Storage`; `Grad []float32` → `Grad *storage.Storage`.
- [ ] Add transitional `func (t *Tensor) Data() []float32 { return t.data.F32() }` (delete in P6).
- [ ] `NewTensor` / `NewEmptyTensor` (`tensor.go:26,64`): build a `Storage` (the `ToDevice` host→device copy now lives inside `storage`, not here).
- [ ] `AccumulateGrad` (`:176`): `grad *storage.Storage`; reframe the three paths onto `AddS`.
- [ ] `ensureGrad`/`AllocGrad` (`:320,:332`): allocate a zeroed `Storage` of matching dtype.
- [ ] Leave `Free`/`ClearGraph`/`ClearComputationGraph` in place for now (deleted in P4).
- [ ] `go build ./...` → fix the call-site list it prints, file by file.
- [ ] `tensor/tests/leak_test.go`: N steps, assert `dev.Stats().BytesInUse` returns to baseline. **This passing is the gate to start Phase 4.**
