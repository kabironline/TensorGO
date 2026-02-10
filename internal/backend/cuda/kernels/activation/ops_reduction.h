#ifndef OPS_REDUCTION_H
#define OPS_REDUCTION_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Reduces entire array to a single sum
float cuda_sum(float* input, int size, cudaStream_t stream);

// Reduces entire array to a single mean
float cuda_mean(float* input, int size, cudaStream_t stream);

// Sums along a specific axis into caller-allocated output buffer
void cuda_sum_axis(float* input, int* shape, int ndim, int axis, float* output, int out_size, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // OPS_REDUCTION_H
