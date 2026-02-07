package cuda

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcuda -lcublas -L${SRCDIR}/kernels -lcuda
#cgo CFLAGS: -I/usr/local/cuda/include -I${SRCDIR} -I${SRCDIR}/kernels -I${SRCDIR}/kernels/optim
#include <cuda_runtime.h>
#include "ops_optim.h"
*/
import "C"
import "unsafe"

// StepSGD performs a single step of SGD optimization on the GPU
// data:   parameters to update
// grad:   gradients
// lr:     learning rate
func (bk *CUDABackend) StepSGD(data, grad []float32, lr float32) {
	if bk == nil || bk.stream == nil {
		panic("CUDA backend not initialized")
	}

	if len(data) == 0 || len(grad) == 0 {
		panic("invalid tensor pointers")
	}

	if len(data) != len(grad) {
		panic("data and grad must have the same size")
	}

	ret := C.cuda_step_sgd(
		(*C.float)(unsafe.Pointer(&data[0])),
		(*C.float)(unsafe.Pointer(&grad[0])),
		C.float(lr),
		C.int(len(data)),
		C.cudaStream_t(bk.stream),
	)

	if ret != 0 {
		panic("cuda_step_sgd failed")
	}
}

// StepAdam performs a single step of Adam optimization on the GPU
// data:   parameters to update
// grad:   gradients
// m:      first moment estimates (mean of gradients)
// v:      second moment estimates (mean of squared gradients)
// lr:     learning rate (typically 0.001)
// beta1:  exponential decay rate for first moment (default 0.9)
// beta2:  exponential decay rate for second moment (default 0.999)
// eps:    small constant for numerical stability (default 1e-8)
// t:      timestep (1-indexed)
func (bk *CUDABackend) StepAdam(data, grad, m, v []float32, lr, beta1, beta2, eps float32, t int) {
	if bk == nil || bk.stream == nil {
		panic("CUDA backend not initialized")
	}

	if len(data) == 0 || len(grad) == 0 || len(m) == 0 || len(v) == 0 {
		panic("invalid tensor pointers")
	}

	size := len(data)
	if len(grad) != size || len(m) != size || len(v) != size {
		panic("all tensors must have the same size")
	}

	if t <= 0 {
		panic("timestep t must be positive")
	}

	ret := C.cuda_step_adam(
		(*C.float)(unsafe.Pointer(&data[0])),
		(*C.float)(unsafe.Pointer(&grad[0])),
		(*C.float)(unsafe.Pointer(&m[0])),
		(*C.float)(unsafe.Pointer(&v[0])),
		C.float(lr),
		C.float(beta1),
		C.float(beta2),
		C.float(eps),
		C.int(t),
		C.int(size),
		C.cudaStream_t(bk.stream),
	)

	if ret != 0 {
		panic("cuda_step_adam failed")
	}
}
