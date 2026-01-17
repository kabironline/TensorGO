#ifndef DMAS_OPS_H
#define DMAS_OPS_H
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

// Element-wise addition: out[i] = a[i] + b[i]
int cuda_add(float *d_a, float *d_b, float *out,
                 int size, cudaStream_t stream, cublasHandle_t handle);

// Element-wise subtraction: out[i] = a[i] - b[i]
int cuda_sub(float *d_a, float *d_b, float *out,
                 int size, cudaStream_t stream, cublasHandle_t handle);

int cuda_mul(float *d_a, float *d_b, float *out,int size, 
                 cudaStream_t stream);

int cuda_div(float *d_a, float *d_b, float *out,
                 int size, cudaStream_t stream);
                 
#ifdef __cplusplus
}
#endif

#endif