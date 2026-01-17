package cuda

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcuda -lcudart -lcublas -L${SRCDIR}/kernels -lcuda
#cgo CFLAGS: -I/usr/local/cuda/include -I${SRCDIR}/kernels/dmas
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "ops_dmas.h"
*/
import "C"
import "unsafe"

func (bk *CUDABackend) Add(d_a, d_b, d_out []float32, n int) {
	// when tensors are allocated they are directly mapped to GPU memory
	// No need to copy data to GPU memory
	if bk == nil || bk.cuBLASHandle == nil {
		panic("cuBLAS handle not initialized")
	}

	if len(d_a) == 0 || len(d_b) == 0 || len(d_a) != len(d_b) {
		panic("invalid matrix pointers")
	}

	ret := C.cuda_add(
		(*C.float)(unsafe.Pointer(&d_a[0])),
		(*C.float)(unsafe.Pointer(&d_b[0])),
		(*C.float)(unsafe.Pointer(&d_out[0])),
		C.int(n),
		C.cudaStream_t(bk.stream),
		C.cublasHandle_t(bk.cuBLASHandle),
	)

	if ret != 0 {
		panic("cuda_add failed")
	}

}
