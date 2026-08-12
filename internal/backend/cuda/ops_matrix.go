package cuda

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcuda -lcublas -L${SRCDIR}/kernels -lcuda
#cgo CFLAGS: -I/usr/local/cuda/include -I${SRCDIR} -I${SRCDIR}/kernels -I${SRCDIR}/kernels/matrix
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "ops_matrix.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/kabironline/nanograd/backend"
)

// MatMul computes out = alpha*(A @ B) + beta*out.
//
// The existing kernels come in three fixed forms (no-trans, trans-A, trans-B)
// and do not take alpha/beta, so this dispatches on the operands' Trans flags
// and rejects anything the kernels cannot express. Those cases are reachable
// only from code paths that don't exist yet; when they do, add a kernel rather
// than silently computing something else.
func (bk *CUDABackend) MatMul(a, b backend.MatOperand, out []float32, alpha, beta float32) {
	if bk == nil || bk.cuBLASHandle == nil {
		panic("cuBLAS handle not initialized")
	}
	if len(a.Data) == 0 || len(b.Data) == 0 || len(out) == 0 {
		panic("MatMul: empty operand")
	}
	if alpha != 1.0 || beta != 0.0 {
		panic(fmt.Sprintf(
			"MatMul: CUDA kernels only implement alpha=1, beta=0 (got alpha=%v, beta=%v)",
			alpha, beta))
	}

	// Logical dimensions of the product: (a.Rows x a.Cols) @ (b.Rows x b.Cols).
	if a.Cols != b.Rows {
		panic(fmt.Sprintf("MatMul: inner dimensions disagree: %d vs %d", a.Cols, b.Rows))
	}
	m, n, k := a.Rows, b.Cols, a.Cols

	pa := (*C.float)(unsafe.Pointer(&a.Data[0]))
	pb := (*C.float)(unsafe.Pointer(&b.Data[0]))
	po := (*C.float)(unsafe.Pointer(&out[0]))
	cm, cn, ck := C.int(m), C.int(n), C.int(k)
	lda, ldb := C.int(a.LD), C.int(b.LD)
	h := C.cublasHandle_t(bk.cuBLASHandle)

	switch {
	case !a.Trans && !b.Trans:
		if ret := C.cuda_matmul(pa, pb, po, cm, cn, ck, lda, ldb, h); ret != 0 {
			panic(fmt.Sprintf("cuda_matmul failed: %d", int(ret)))
		}

	// NOTE: cuda_matmul_trans_a/_trans_b are declared `void` in
	// kernels/matrix/ops_matrix.h — they cannot report a cuBLAS failure at all,
	// so a bad launch here surfaces later as a sticky error somewhere unrelated.
	// Give them an int return and check it (P3).
	case a.Trans && !b.Trans:
		C.cuda_matmul_trans_a(pa, pb, po, cm, cn, ck, lda, ldb, h)

	case !a.Trans && b.Trans:
		C.cuda_matmul_trans_b(pa, pb, po, cm, cn, ck, lda, ldb, h)

	default:
		panic("MatMul: no CUDA kernel for both operands transposed")
	}
}
