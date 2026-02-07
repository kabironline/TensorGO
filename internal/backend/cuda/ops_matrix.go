package cuda

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcuda -lcublas -L${SRCDIR}/kernels -lcuda
#cgo CFLAGS: -I/usr/local/cuda/include -I${SRCDIR} -I${SRCDIR}/kernels -I${SRCDIR}/kernels/matrix
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "ops_matrix.h"
*/
import "C"

import "unsafe"

func (bk *CUDABackend) MatMul(d_a, d_b, out []float32, m, n, k, sA, sB int) []float32 {
	// when tensors are allocated they are directly mapped to GPU memory
	// No need to copy data to GPU memory
	if bk == nil || bk.cuBLASHandle == nil {
		panic("cuBLAS handle not initialized")
	}

	if len(d_a) == 0 || len(d_b) == 0 || len(out) == 0 {
		panic("invalid matrix pointers")
	}

	// Call the kernel wrapper implemented in the CUDA module (./kernels/matrix/mul.cu)
	ret := C.cuda_matmul(
		(*C.float)(unsafe.Pointer(&d_a[0])),
		(*C.float)(unsafe.Pointer(&d_b[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(m), C.int(n), C.int(k), C.int(sA), C.int(sB),
		C.cublasHandle_t(bk.cuBLASHandle),
	)

	if ret != 0 {
		panic("cuda_matmul failed")
	}
	return out
}

func (b *CUDABackend) MatMulTransA(a, b_val, out []float32, m, n, k, strideA, strideB int) []float32 {
	C.cuda_matmul_trans_a(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b_val[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(m), C.int(n), C.int(k),
		C.int(strideA), C.int(strideB),
		C.cublasHandle_t(b.cuBLASHandle),
	)
	return out
}

func (b *CUDABackend) MatMulTransB(a, b_val, out []float32, m, n, k, strideA, strideB int) []float32 {
	C.cuda_matmul_trans_b(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b_val[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(m), C.int(n), C.int(k),
		C.int(strideA), C.int(strideB),
		C.cublasHandle_t(b.cuBLASHandle),
	)
	return out
}
