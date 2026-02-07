#ifndef OPS_MATRIX_H
#define OPS_MATRIX_H
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

int cuda_matmul(float *d_a, float *d_b, float *out,
                 int m, int n, int k, int sA, int sB, cublasHandle_t handle);

void cuda_matmul_trans_a(const float* A, const float* B, float* C,
                         int m, int n, int k, int strideA, int strideB, cublasHandle_t handle);

void cuda_matmul_trans_b(const float* A, const float* B, float* C,
                         int m, int n, int k, int strideA, int strideB, cublasHandle_t handle);

#ifdef __cplusplus
}
#endif

#endif