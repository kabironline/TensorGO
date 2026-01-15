package cuda

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcudart
#include <cuda_runtime.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// ToDevice copies data from CPU (host) to GPU (device)
func (b *CUDABackend) ToDevice(data []float64) []float64 {
	size := len(data)
	if size == 0 {
		return nil
	}

	// 1. Allocate GPU memory via our pool
	gpuData := b.Allocate(size)

	// 2. Perform Synchronous Copy (Host to Device)
	// We use unsafe.SliceData to get the pointer without dereferencing a single element,
	// which prevents page faults when Go tries to "check" the element.
	res := C.cudaMemcpyAsync(
		unsafe.Pointer(unsafe.SliceData(gpuData)),
		unsafe.Pointer(unsafe.SliceData(data)),
		C.size_t(size*8), // 8 bytes per float64
		C.cudaMemcpyHostToDevice,
		C.cudaStream_t(b.stream),
	)

	if res != C.cudaSuccess {
		panic(fmt.Sprintf("ToDevice copy failed: %v", res))
	}

	return gpuData
}

// ToCPU copies data from GPU (device) back to CPU (host)
func (b *CUDABackend) ToCPU(data []float64) []float64 {
	size := len(data)
	if size == 0 {
		return nil
	}

	// 1. Allocate standard Go slice on CPU
	hostData := make([]float64, size)

	// 2. Perform Synchronous Copy (Device to Host)
	res := C.cudaMemcpyAsync(
		unsafe.Pointer(unsafe.SliceData(hostData)),
		unsafe.Pointer(unsafe.SliceData(data)),
		C.size_t(size*8),
		C.cudaMemcpyDeviceToHost,
		C.cudaStream_t(b.stream),
	)

	if res != C.cudaSuccess {
		panic(fmt.Sprintf("ToCPU copy failed: %v", res))
	}

	return hostData
}

func (b *CUDABackend) Allocate(size int) []float64 {
	if size == 0 {
		return nil
	}
	ptr, err := b.memPool.Allocate(size * 8) // size of float64
	if err != nil {
		panic(err)
	}
	// Use unsafe.Slice for a safer and more modern way to create the slice header
	return unsafe.Slice((*float64)(ptr), size)
}

func (b *CUDABackend) Free(data []float64) {
	if len(data) == 0 {
		return
	}
	ptr := unsafe.Pointer(unsafe.SliceData(data))
	if err := b.memPool.Free(ptr); err != nil {
		panic(err)
	}
}

func (b *CUDABackend) Copy(dst, src []float64) {
	size := len(src)
	if size == 0 {
		return
	}
	res := C.cudaMemcpy(
		unsafe.Pointer(unsafe.SliceData(dst)),
		unsafe.Pointer(unsafe.SliceData(src)),
		C.size_t(size*8),
		C.cudaMemcpyDeviceToDevice,
	)
	if res != C.cudaSuccess {
		panic(fmt.Sprintf("Device-to-device copy failed: %v", res))
	}
}

func (b *CUDABackend) Sync() {
	C.cudaDeviceSynchronize()
}
