#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Copy a possibly non-contiguous source buffer into a contiguous destination.
// h_shape and h_strides point to host arrays of length `ndim`.
int cuda_contiguous(float *d_src, float *d_dst, int *h_shape, int *h_strides, int ndim, int total, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
