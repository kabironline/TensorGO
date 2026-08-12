package cuda

// Phase-2 Storage-typed memory management for the CUDA backend.
//
// NOTE: this file is pure Go (no direct cgo) — it goes through the existing
// memory pool and b.Copy — but it lives in the cgo `cuda` package, so it only
// builds with CGO_ENABLED=1 and the CUDA toolchain present. CUDA is F32-only for
// now (see CopyStorage), matching the rest of the backend.

import (
	"unsafe"

	"github.com/kabironline/nanograd/storage"
)

// gpuBuffer is a device-resident buffer. Its bytes are NOT addressable from Go,
// so Bytes() panics — callers must copy to host first. Free returns the block to
// the pool (deterministic, unlike the old SetFinalizer path).
type gpuBuffer struct {
	ptr  unsafe.Pointer
	size int // bytes
	pool *MemoryPool
}

func (g *gpuBuffer) Bytes() []byte {
	panic("cuda: Bytes() called on device memory — copy to host (ToCPU) first")
}

func (g *gpuBuffer) Len() int { return g.size }

func (g *gpuBuffer) Free() {
	if g.ptr == nil {
		return
	}
	if err := g.pool.Free(g.ptr); err != nil {
		panic(err)
	}
	g.ptr = nil
}

// Ptr exposes the raw device pointer for cgo calls elsewhere in the package.
func (g *gpuBuffer) Ptr() unsafe.Pointer { return g.ptr }

// AllocStorage allocates numel elements of dtype dt as device memory from the pool.
func (b *CUDABackend) AllocStorage(numel int, dt storage.DType) *storage.Storage {
	if numel == 0 {
		return storage.New(&gpuBuffer{pool: b.memPool}, dt, 0)
	}
	bytes := numel * dt.Size()
	ptr, err := b.memPool.Allocate(bytes)
	if err != nil {
		panic(err)
	}
	return storage.New(&gpuBuffer{ptr: ptr, size: bytes, pool: b.memPool}, dt, numel)
}

// FreeStorage returns the storage's device buffer to the pool.
func (b *CUDABackend) FreeStorage(s *storage.Storage) {
	if s != nil {
		s.Free()
	}
}

// CopyStorage performs a device-to-device copy. CUDA is F32-only for now, so the
// element count maps directly onto b.Copy's []float32 view of the device buffers.
func (b *CUDABackend) CopyStorage(dst, src *storage.Storage) {
	n := src.Length()
	if n == 0 {
		return
	}
	srcB := src.Buffer().(*gpuBuffer)
	dstB := dst.Buffer().(*gpuBuffer)
	srcF := unsafe.Slice((*float32)(srcB.ptr), n)
	dstF := unsafe.Slice((*float32)(dstB.ptr), n)
	b.Copy(dstF, srcF)
}
