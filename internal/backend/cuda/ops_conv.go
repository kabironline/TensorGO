package cuda

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcuda -lcublas -lcudnn -L${SRCDIR}/kernels -lcuda
#cgo CFLAGS: -I/usr/local/cuda/include -I${SRCDIR} -I${SRCDIR}/kernels -I${SRCDIR}/kernels/conv
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include "ops_conv.h"
*/
import "C"
import "unsafe"

func (b *CUDABackend) Conv2DForward(
	input, weights, bias []float32,
	batchSize, inChannels, inHeight, inWidth int,
	outChannels, kernelHeight, kernelWidth int,
	padHeight, padWidth int,
	strideHeight, strideWidth int,
) []float32 {
	if b == nil || b.stream == nil || b.cuDNNHandle == nil {
		panic("cuDNN handle not initialized")
	}
	if len(input) == 0 || len(weights) == 0 {
		return nil
	}

	outHeight := (inHeight+2*padHeight-kernelHeight)/strideHeight + 1
	outWidth := (inWidth+2*padWidth-kernelWidth)/strideWidth + 1
	if outHeight <= 0 || outWidth <= 0 {
		return nil
	}

	outSize := batchSize * outChannels * outHeight * outWidth
	if outSize <= 0 {
		return nil
	}

	out := b.Allocate(outSize)
	if len(out) == 0 {
		return nil
	}

	var biasPtr *C.float
	if len(bias) > 0 {
		biasPtr = (*C.float)(unsafe.Pointer(&bias[0]))
	}

	ret := C.cuda_convolution_forward(
		(*C.float)(unsafe.Pointer(&input[0])),
		(*C.float)(unsafe.Pointer(&weights[0])),
		biasPtr,
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(batchSize), C.int(inChannels), C.int(inHeight), C.int(inWidth),
		C.int(outChannels), C.int(kernelHeight), C.int(kernelWidth),
		C.int(padHeight), C.int(padWidth),
		C.int(strideHeight), C.int(strideWidth),
		C.cudaStream_t(b.stream),
		C.cudnnHandle_t(b.cuDNNHandle),
	)
	if ret != 0 {
		panic("cuda_convolution_forward failed")
	}
	return out
}

func (b *CUDABackend) Conv2DBackward(
	input, weights, outputGrad []float32,
	batchSize, inChannels, inHeight, inWidth int,
	outChannels, kernelHeight, kernelWidth int,
	padHeight, padWidth int,
	strideHeight, strideWidth int,
) (inputGrad, weightsGrad, biasGrad []float32) {
	if b == nil || b.stream == nil || b.cuDNNHandle == nil {
		panic("cuDNN handle not initialized")
	}
	if len(input) == 0 || len(weights) == 0 || len(outputGrad) == 0 {
		return nil, nil, nil
	}

	inSize := batchSize * inChannels * inHeight * inWidth
	if inSize <= 0 {
		return nil, nil, nil
	}
	weightSize := outChannels * inChannels * kernelHeight * kernelWidth
	if weightSize <= 0 {
		return nil, nil, nil
	}

	inputGrad = b.Allocate(inSize)
	weightsGrad = b.Allocate(weightSize)
	biasGrad = b.Allocate(outChannels)

	ret := C.cuda_convolution_backward(
		(*C.float)(unsafe.Pointer(&input[0])),
		(*C.float)(unsafe.Pointer(&weights[0])),
		(*C.float)(unsafe.Pointer(&outputGrad[0])),
		(*C.float)(unsafe.Pointer(&inputGrad[0])),
		(*C.float)(unsafe.Pointer(&weightsGrad[0])),
		(*C.float)(unsafe.Pointer(&biasGrad[0])),
		C.int(batchSize), C.int(inChannels), C.int(inHeight), C.int(inWidth),
		C.int(outChannels), C.int(kernelHeight), C.int(kernelWidth),
		C.int(padHeight), C.int(padWidth),
		C.int(strideHeight), C.int(strideWidth),
		C.cudaStream_t(b.stream),
		C.cudnnHandle_t(b.cuDNNHandle),
	)
	if ret != 0 {
		panic("cuda_convolution_backward failed")
	}
	return inputGrad, weightsGrad, biasGrad
}
