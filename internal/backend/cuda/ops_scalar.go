package cuda

/*
#cgo CFLAGS: -I${SRCDIR}/kernels/activation
#cgo LDFLAGS: -L${SRCDIR}/kernels -lcuda -lcublas -lcudart -lm

#include "ops_scalar.h"
*/
import "C"
import "unsafe"

func (b *CUDABackend) AddScalar(a []float32, scalar float32, size int) []float32 {
	out := b.Allocate(size)
	C.cuda_add_scalar(
		(*C.float)(unsafe.Pointer(&a[0])),
		C.float(scalar),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(size),
		C.cudaStream_t(b.stream),
	)
	return out
}

func (b *CUDABackend) SubScalar(a []float32, scalar float32, size int) []float32 {
	out := b.Allocate(size)
	C.cuda_sub_scalar(
		(*C.float)(unsafe.Pointer(&a[0])),
		C.float(scalar),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(size),
		C.cudaStream_t(b.stream),
	)
	return out
}

func (b *CUDABackend) MulScalar(a []float32, scalar float32, size int) []float32 {
	out := b.Allocate(size)
	C.cuda_mul_scalar(
		(*C.float)(unsafe.Pointer(&a[0])),
		C.float(scalar),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(size),
		C.cudaStream_t(b.stream),
	)
	return out
}

func (b *CUDABackend) DivScalar(a []float32, scalar float32, size int) []float32 {
	out := b.Allocate(size)
	C.cuda_div_scalar(
		(*C.float)(unsafe.Pointer(&a[0])),
		C.float(scalar),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(size),
		C.cudaStream_t(b.stream),
	)
	return out
}
