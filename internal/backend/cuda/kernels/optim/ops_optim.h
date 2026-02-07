#ifndef OPS_OPTIM_H
#define OPS_OPTIM_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Performs a single step of SGD update: data -= lr * grad
int cuda_step_sgd(float *data, const float *grad, float lr, int size, cudaStream_t stream);

// Performs a single step of Adam update
// data:   parameters to update
// grad:   gradients
// m:      first moment estimates (mean of gradients)
// v:      second moment estimates (mean of squared gradients)
// lr:     learning rate
// beta1:  exponential decay rate for first moment (default 0.9)
// beta2:  exponential decay rate for second moment (default 0.999)
// eps:    small constant for numerical stability (default 1e-8)
// t:      timestep (1-indexed)
// size:   number of elements
int cuda_step_adam(float *data, const float *grad, float *m, float *v,
                   float lr, float beta1, float beta2, float eps, int t, int size, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
