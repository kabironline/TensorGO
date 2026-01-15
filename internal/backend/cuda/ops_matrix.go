package cuda

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcuda  -lcublas
#cgo CFLAGS: -I/usr/local/cuda/include
#include <cuda_runtime.h>
#include <cublas_v2.h>
*/
import "C"

import "unsafe"

func (bk *CUDABackend) MatMul(d_a, d_b, out []float32, m, n, k, sA, sB int) []float32 {
	// when tensors are allocated they are directly mapped to GPU memory
	// No need to copy data to GPU memory
	if bk == nil || bk.cuBLASHandle == nil {
		panic("cuBLAS handle not initialized")
	}
	handle := C.cublasHandle_t(bk.cuBLASHandle)

	alphaC := C.float(1.0)
	betaC := C.float(0.0)

	if len(d_a) == 0 || len(d_b) == 0 || len(out) == 0 {
		panic("invalid matrix pointers")
	}

	// Note the order of supplying A and B are reversed to compute (A@B)^T
	// This is because cuBLAS expects column-major order for matrices
	// and we store everything in row-major order
	// We keep both operands non-transposed and swap A/B, then swap m/n.
	res := C.cublasSgemm(handle,
		C.cublasOperation_t(0), C.cublasOperation_t(0),
		C.int(n), C.int(m), C.int(k),
		&alphaC,
		(*C.float)(unsafe.Pointer(&d_b[0])), C.int(sB),
		(*C.float)(unsafe.Pointer(&d_a[0])), C.int(sA),
		&betaC,
		(*C.float)(unsafe.Pointer(&out[0])), C.int(n),
	)

	if res != C.CUBLAS_STATUS_SUCCESS {
		panic("Failed to perform cublas sgemm")
	}
	return out
}
