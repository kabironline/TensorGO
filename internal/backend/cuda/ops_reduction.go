package cuda

/*
#cgo CFLAGS: -I${SRCDIR}/kernels/activation
#cgo LDFLAGS: -L${SRCDIR}/kernels -lcuda -lcublas -lcudart -lm

#include "ops_reduction.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

func (b *CUDABackend) Sum(data []float32, size int) float32 {
	if size == 0 {
		return 0.0
	}
	result := C.cuda_sum(
		(*C.float)(unsafe.Pointer(&data[0])),
		C.int(size),
		C.cudaStream_t(b.stream),
	)
	return float32(result)
}

func (b *CUDABackend) Mean(data []float32, size int) float32 {
	if size == 0 {
		return 0.0
	}
	result := C.cuda_mean(
		(*C.float)(unsafe.Pointer(&data[0])),
		C.int(size),
		C.cudaStream_t(b.stream),
	)
	return float32(result)
}

func (b *CUDABackend) SumAxis(data []float32, shape []int, axis int) []float32 {
	if axis < 0 || axis >= len(shape) {
		panic("SumAxis: invalid axis")
	}

	// Convert shape to C array
	cShape := make([]C.int, len(shape))
	for i, s := range shape {
		cShape[i] = C.int(s)
	}

	// Compute output size: product of dims excluding axis
	outSize := 1
	for i, s := range shape {
		if i == axis {
			continue
		}
		outSize *= s
	}
	if outSize == 0 {
		return nil
	}

	result := b.Allocate(outSize)
	C.cuda_sum_axis(
		(*C.float)(unsafe.Pointer(&data[0])),
		(*C.int)(unsafe.Pointer(&cShape[0])),
		C.int(len(shape)),
		C.int(axis),
		(*C.float)(unsafe.Pointer(&result[0])),
		C.int(outSize),
		C.cudaStream_t(b.stream),
	)

	return result
}
