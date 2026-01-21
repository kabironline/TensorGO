#ifndef OPS_ACTIVATION_H
#define OPS_ACTIVATION_H
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ReLU activation: out[i] = max(0, in[i])
int cuda_relu(float *d_in, float *out,
                 int size, cudaStream_t stream);

// Relu Backward: out[i] = in_grad[i] * (in[i] > 0 ? 1 : 0)
int cuda_relu_backward(float *d_in, float *d_in_grad, float *out,
                 int size, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif