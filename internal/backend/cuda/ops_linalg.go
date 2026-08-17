//go:build cuda

package cuda

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcudart -lcublas -L${SRCDIR}/kernels -lcuda
#cgo CFLAGS: -I/usr/local/cuda/include -I${SRCDIR}/kernels -I${SRCDIR}/kernels/linalg
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "ops_linalg.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// ErrSingularMatrix is returned when a matrix has no inverse. It is a property
// of the data, not a programmer mistake, so it is reported rather than panicked.
var ErrSingularMatrix = errors.New("cuda: matrix is singular and cannot be inverted")

// InverseSmallMax is the largest n handled by the shared-memory Gauss-Jordan
// path; above it Inverse switches to cuBLAS batched LU. Exposed so tests and
// benchmarks can straddle the boundary deliberately.
const InverseSmallMax = int(C.CUDA_INVERSE_SMALL_MAX)

// Inverse writes the inverse of the n x n row-major matrix a into out.
//
// a and out are device buffers and must not alias; a is not modified. The
// implementation picks a strategy based on n -- see inverseWithPath.
func (bk *CUDABackend) Inverse(a, out []float32, n int) error {
	return bk.inverseWithPath(a, out, n, pathAuto)
}

type inversePath int

const (
	pathAuto inversePath = iota
	pathSmall
	pathLarge
)

// inverseWithPath is Inverse with the dispatch decision forced. Only tests and
// benchmarks pass anything other than pathAuto -- comparing the two strategies
// at the same n is the only way to know the crossover is in the right place.
func (bk *CUDABackend) inverseWithPath(a, out []float32, n int, path inversePath) error {
	if bk == nil || bk.stream == nil {
		return errors.New("cuda: backend not initialized")
	}
	if n <= 0 {
		return fmt.Errorf("Inverse: n must be positive, got %d", n)
	}
	if len(a) < n*n {
		return fmt.Errorf("Inverse: input has %d elements, need %d", len(a), n*n)
	}
	if len(out) < n*n {
		return fmt.Errorf("Inverse: out has %d elements, need %d", len(out), n*n)
	}
	if path == pathSmall && n > InverseSmallMax {
		return fmt.Errorf("Inverse: small path supports n <= %d, got %d", InverseSmallMax, n)
	}

	pa := (*C.float)(unsafe.Pointer(&a[0]))
	po := (*C.float)(unsafe.Pointer(&out[0]))
	cn := C.int(n)
	stream := C.cudaStream_t(bk.stream)

	var rc C.int
	switch path {
	case pathSmall:
		rc = C.cuda_inverse_small(pa, po, cn, stream)
	case pathLarge:
		rc = C.cuda_inverse_large(pa, po, cn, C.cublasHandle_t(bk.cuBLASHandle), stream)
	default:
		rc = C.cuda_inverse(pa, po, cn, C.cublasHandle_t(bk.cuBLASHandle), stream)
	}

	switch rc {
	case 0:
		return nil
	case 1:
		return ErrSingularMatrix
	default:
		return fmt.Errorf("cuda_inverse failed for n=%d (code %d)", n, int(rc))
	}
}

// InverseSmallForTest and InverseLargeForTest force a specific inversion
// strategy regardless of n.
//
// They exist because the CUDA tests live in a separate package
// (internal/backend/cuda/test), so an export_test.go hook is not visible to
// them. Comparing both strategies at the same n is the only way to check they
// agree and to justify where the crossover sits. Not part of the supported API.
func InverseSmallForTest(bk *CUDABackend, a, out []float32, n int) error {
	return bk.inverseWithPath(a, out, n, pathSmall)
}

func InverseLargeForTest(bk *CUDABackend, a, out []float32, n int) error {
	return bk.inverseWithPath(a, out, n, pathLarge)
}
