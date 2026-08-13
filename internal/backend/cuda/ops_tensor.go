//go:build cuda

package cuda

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcuda -lcudart -L${SRCDIR}/kernels -lcuda
#cgo CFLAGS: -I/usr/local/cuda/include -I${SRCDIR}/kernels/tensor
#include <cuda_runtime.h>
#include "ops_tensor.h"
*/
import "C"
import "unsafe"

// Contiguous: GPU implementation that writes a contiguous copy of `d_src` into `d_dst`.
func (bk *CUDABackend) Contiguous(d_src, d_dst []float32, shape, strides []int, offset int) {
	if bk == nil || bk.stream == nil {
		panic("CUDA stream not initialized")
	}
	if len(shape) == 0 || len(strides) == 0 || len(shape) != len(strides) {
		panic("invalid shape/strides")
	}
	total := 1
	for _, s := range shape {
		total *= s
	}

	ret := C.cuda_contiguous(
		(*C.float)(unsafe.Pointer(&d_src[0])),
		(*C.float)(unsafe.Pointer(&d_dst[0])),
		(*C.int)(unsafe.Pointer(&shape[0])),
		(*C.int)(unsafe.Pointer(&strides[0])),
		C.int(len(shape)),
		C.int(total),
		C.int(offset),
		C.cudaStream_t(bk.stream),
	)

	if ret != 0 {
		panic("cuda_contiguous failed")
	}
}
