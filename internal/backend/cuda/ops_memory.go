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
func (b *CUDABackend) ToDevice(data []float32) []float32 {
	size := len(data)
	if size == 0 {
		return nil
	}

	// 1. Allocate GPU memory via our pool
	gpuData := b.Allocate(size)

	// 2. Perform asynchronous copy (Host to Device)
	// We use unsafe.SliceData to get the pointer without dereferencing a single element.
	// Each float32 is 4 bytes, so multiply by 4 to get the full byte size.
	res := C.cudaMemcpyAsync(
		unsafe.Pointer(unsafe.SliceData(gpuData)),
		unsafe.Pointer(unsafe.SliceData(data)),
		C.size_t(size*4), // 4 bytes per float32
		C.cudaMemcpyHostToDevice,
		C.cudaStream_t(b.stream),
	)

	if res != C.cudaSuccess {
		panic(fmt.Sprintf("ToDevice copy failed: %v", res))
	}

	return gpuData
}

// ToCPU copies data from GPU (device) back to CPU (host)
func (b *CUDABackend) ToCPU(data []float32) []float32 {
	size := len(data)
	if size == 0 {
		return nil
	}

	// 1. Allocate standard Go slice on CPU
	hostData := make([]float32, size)

	// 2. Perform asynchronous copy (Device to Host)
	// Each float32 is 4 bytes.
	res := C.cudaMemcpyAsync(
		unsafe.Pointer(unsafe.SliceData(hostData)),
		unsafe.Pointer(unsafe.SliceData(data)),
		C.size_t(size*4),
		C.cudaMemcpyDeviceToHost,
		C.cudaStream_t(b.stream),
	)

	if res != C.cudaSuccess {
		panic(fmt.Sprintf("ToCPU copy failed: %v", res))
	}

	return hostData
}

// WriteToDevice copies data from a host slice into an existing device buffer `dst`.
// `dst` must be a device slice previously allocated by Allocate().
func (b *CUDABackend) WriteToDevice(dst []float32, src []float32) {
	size := len(src)
	if size == 0 {
		return
	}
	if len(dst) != size {
		panic("WriteToDevice: destination length mismatch")
	}

	res := C.cudaMemcpyAsync(
		unsafe.Pointer(unsafe.SliceData(dst)),
		unsafe.Pointer(unsafe.SliceData(src)),
		C.size_t(size*4),
		C.cudaMemcpyHostToDevice,
		C.cudaStream_t(b.stream),
	)

	if res != C.cudaSuccess {
		panic(fmt.Sprintf("WriteToDevice failed: %v", res))
	}
}

func (b *CUDABackend) Allocate(size int) []float32 {
	if size == 0 {
		return nil
	}
	ptr, err := b.memPool.Allocate(size * 4) // size of float32 (bytes)
	if err != nil {
		panic(err)
	}
	// Use unsafe.Slice for a safer and more modern way to create the slice header
	return unsafe.Slice((*float32)(ptr), size)
}

func (b *CUDABackend) Free(data []float32) {
	if len(data) == 0 {
		return
	}
	ptr := unsafe.Pointer(unsafe.SliceData(data))
	if err := b.memPool.Free(ptr); err != nil {
		panic(err)
	}
}

func (b *CUDABackend) Copy(dst, src []float32) {
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
