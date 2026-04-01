package cuda

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcudart -lcuda -lcublas -lstdc++ -L${SRCDIR}/kernels -lcuda
#cgo CFLAGS: -I/usr/local/cuda/include -I${SRCDIR} -I${SRCDIR}/kernels -I${SRCDIR}/kernels/activation
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "ops_activation.h"
*/
import "C"
import "unsafe"

func (bk *CUDABackend) ReLU(d_in, d_out []float32, n int) {
	// when tensors are allocated they are directly mapped to GPU memory
	// No need to copy data to GPU memory
	if bk == nil || bk.stream == nil {
		panic("stream not initialized")
	}

	if len(d_in) == 0 {
		panic("invalid matrix pointer")
	}

	ret := C.cuda_relu(
		(*C.float)(unsafe.Pointer(&d_in[0])),
		(*C.float)(unsafe.Pointer(&d_out[0])),
		C.int(n),
		C.cudaStream_t(bk.stream),
	)

	if ret != 0 {
		panic("cuda_relu failed")
	}
}

func (bk *CUDABackend) ReLUBackward(d_grad, d_in, d_out []float32, n int) {
	// when tensors are allocated they are directly mapped to GPU memory
	// No need to copy data to GPU memory
	if bk == nil || bk.stream == nil {
		panic("stream not initialized")
	}

	if len(d_in) == 0 || len(d_grad) == 0 {
		panic("invalid matrix pointer")
	}

	ret := C.cuda_relu_backward(
		(*C.float)(unsafe.Pointer(&d_grad[0])),
		(*C.float)(unsafe.Pointer(&d_in[0])),
		(*C.float)(unsafe.Pointer(&d_out[0])),
		C.int(n),
		C.cudaStream_t(bk.stream),
	)

	if ret != 0 {
		panic("cuda_relu_backward failed")
	}
}
