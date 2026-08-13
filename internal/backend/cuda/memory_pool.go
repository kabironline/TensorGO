//go:build cuda

package cuda

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcuda -lcudart
#cgo CFLAGS: -I/usr/local/cuda/include
#include <cuda_runtime.h>
*/
import "C"

import (
	"fmt"
	"math/bits"
	"sync"
	"unsafe"
)

type Block struct {
	ptr  unsafe.Pointer
	size int // The actual size allocated (power of 2)
}

type MemoryPool struct {
	mu sync.Mutex

	// buckets[pow2_exponent] -> list of free blocks
	// e.g., buckets[10] holds blocks of size 1024
	buckets [64][]Block

	// Tracks all blocks currently out in the wild
	activeBlocks map[uintptr]int // ptr -> size

	currentPoolSize int64
	maxPoolSize     int64
}

func NewMemoryPool(maxSize int64) *MemoryPool {
	return &MemoryPool{
		activeBlocks: make(map[uintptr]int),
		maxPoolSize:  maxSize,
	}
}

// Allocate gets a memory block. It rounds the size to the next power of 2.
func (p *MemoryPool) Allocate(size int) (unsafe.Pointer, error) {
	if size == 0 {
		return nil, nil
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	// 1. Calculate the bucket index (log2)
	roundedSize := nextPowerOf2(size)
	bucketIdx := bits.Len(uint(roundedSize) - 1)

	// 2. Try to find a block in this bucket or larger buckets
	for i := bucketIdx; i < 64; i++ {
		if len(p.buckets[i]) > 0 {
			// Pop last block (LIFO is better for cache locality)
			idx := len(p.buckets[i]) - 1
			block := p.buckets[i][idx]
			p.buckets[i] = p.buckets[i][:idx]

			p.activeBlocks[uintptr(block.ptr)] = block.size
			return block.ptr, nil
		}
	}

	// 3. Cache miss: Allocate new memory from CUDA
	if p.maxPoolSize > 0 && p.currentPoolSize+int64(roundedSize) > p.maxPoolSize {
		return nil, fmt.Errorf("GPU OOM: Pool limit %d reached", p.maxPoolSize)
	}

	ptr, err := cudaMalloc(roundedSize)
	if err != nil {
		return nil, err
	}

	p.currentPoolSize += int64(roundedSize)
	p.activeBlocks[uintptr(ptr)] = roundedSize
	return ptr, nil
}

func (p *MemoryPool) Free(ptr unsafe.Pointer) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	ptrVal := uintptr(ptr)
	size, ok := p.activeBlocks[ptrVal]
	if !ok {
		return fmt.Errorf("pointer not found in pool (double free or foreign pointer)")
	}

	delete(p.activeBlocks, ptrVal)

	bucketIdx := bits.Len(uint(size) - 1)
	p.buckets[bucketIdx] = append(p.buckets[bucketIdx], Block{ptr: ptr, size: size})
	return nil
}

// Clear frees all idle memory currently held in buckets back to the OS/GPU.
func (p *MemoryPool) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()

	for i := 0; i < 64; i++ {
		for _, block := range p.buckets[i] {
			C.cudaFree(block.ptr)
			p.currentPoolSize -= int64(block.size)
		}
		p.buckets[i] = nil
	}
}

// ============================================================================
// Workspace Manager (Essential for cuDNN/cuBLAS)
// ============================================================================

type WorkspaceManager struct {
	mu   sync.Mutex
	ptr  unsafe.Pointer
	size int
}

// Get ensures a buffer of at least 'size' is available.
// It only reallocates if the requested size exceeds the current buffer.
func (w *WorkspaceManager) Get(size int) (unsafe.Pointer, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if size > w.size {
		if w.ptr != nil {
			cudaFree(w.ptr)
		}
		ptr, err := cudaMalloc(size)
		if err != nil {
			return nil, err
		}
		w.ptr = ptr
		w.size = size
	}
	return w.ptr, nil
}

// ============================================================================
// Utilities
// ============================================================================

func nextPowerOf2(n int) int {
	if n <= 0 {
		return 1
	}
	if n&(n-1) == 0 {
		return n
	}
	return 1 << bits.Len(uint(n))
}

// These will be actual CGO calls to cuda_runtime.h
func cudaMalloc(size int) (unsafe.Pointer, error) {
	var ptr unsafe.Pointer
	// Use regular cudaMalloc for device-only memory (much faster, no page faults)
	res := C.cudaMalloc(&ptr, C.size_t(size))
	if res != C.cudaSuccess {
		return nil, fmt.Errorf("cudaMalloc failed for size %d: code %v", size, res)
	}
	return ptr, nil
}

func cudaFree(ptr unsafe.Pointer) error {
	res := C.cudaFree(ptr)
	if res != C.cudaSuccess {
		return fmt.Errorf("cudaFree failed: code %v", res)
	}
	return nil
}
