#ifndef DMAS_OPS_H
#define DMAS_OPS_H
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

// Element-wise addition: out[i] = a[i] + b[i]
// Uses async copy + cublasSaxpy into `out` on provided stream
int cuda_add(float *d_a, float *d_b, float *out,
                 int size, cudaStream_t stream, cublasHandle_t handle);

#ifdef __cplusplus
}
#endif

#endif