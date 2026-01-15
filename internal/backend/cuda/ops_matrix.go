package cuda

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcuda  -lcublas
#cgo CFLAGS: -I/usr/local/cuda/include
#include <cuda_runtime.h>
#include <cublas_v2.h>
*/
import "C"

import "unsafe"

func (bk *CUDABackend) MatMul(d_a, d_b, out []float64, m, n, k, sA, sB int) []float64 {
	// when tensors are allocated they are directly mapped to GPU memory
	// No need to copy data to GPU memory
	if bk == nil || bk.cuBLASHandle == nil {
		panic("cuBLAS handle not initialized")
	}
	handle := C.cublasHandle_t(bk.cuBLASHandle)

	alphaC := C.double(1.0)
	betaC := C.double(0.0)

	if len(d_a) == 0 || len(d_b) == 0 || len(out) == 0 {
		panic("invalid matrix pointers")
	}

	// Note the order of supplying A and B are reversed to compute (A@B)^T
	// This is because cuBLAS expects column-major order for matrices
	// and we store everything in row-major order
	// We keep both operands non-transposed and swap A/B, then swap m/n.
	res := C.cublasDgemm(handle,
		C.cublasOperation_t(0), C.cublasOperation_t(0),
		C.int(n), C.int(m), C.int(k),
		&alphaC,
		(*C.double)(unsafe.Pointer(&d_b[0])), C.int(sB),
		(*C.double)(unsafe.Pointer(&d_a[0])), C.int(sA),
		&betaC,
		(*C.double)(unsafe.Pointer(&out[0])), C.int(n),
	)

	if res != C.CUBLAS_STATUS_SUCCESS {
		panic("Failed to perform cublas Dgemm")
	}
	return out
}
