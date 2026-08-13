//go:build cuda

package cuda

/*
#cgo CFLAGS: -I${SRCDIR}/kernels/activation
#cgo LDFLAGS: -L${SRCDIR}/kernels -lcuda -lcublas -lcudart -lm

#include "ops_unary.h"
*/
import "C"
import "unsafe"

func (b *CUDABackend) Exp(a []float32, size int) []float32 {
	out := b.Allocate(size)
	C.cuda_exp(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(size),
	)
	return out
}

func (b *CUDABackend) Log(a []float32, size int) []float32 {
	out := b.Allocate(size)
	C.cuda_log(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(size),
	)
	return out
}

func (b *CUDABackend) Square(a []float32, size int) []float32 {
	out := b.Allocate(size)
	C.cuda_square(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(size),
	)
	return out
}

func (b *CUDABackend) Neg(a, out []float32, size int) {
	C.cuda_neg(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(size),
	)
}

func (b *CUDABackend) Sqrt(a []float32, size int) []float32 {
	out := b.Allocate(size)
	C.cuda_sqrt(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(size),
	)
	return out
}
