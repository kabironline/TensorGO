#include "ops_matrix.h"
#include <cublas_v2.h>
#include <stdio.h>

int cuda_matmul(float *d_a, float *d_b, float *out, int m, int n, int k, int sA, int sB, cublasHandle_t handle) {
     if (!handle || !d_a || !d_b || !out) {
        printf("cuda_matmul: Invalid pointers\n");
        return -1;
    }

    if (m <= 0 || n <= 0 || k <= 0) {
        printf("cuda_matmul: Invalid dimensions m=%d n=%d k=%d\n", m, n, k);
        return -1;
    }

    if (sA <= 0 || sB <= 0) {
        printf("cuda_matmul: Invalid strides sA=%d sB=%d\n", sA, sB);
        return -1;
    }



    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Row-major C[m×n] = A[m×k] @ B[k×n]
    // cuBLAS uses column-major, so use identity: C^T = B^T @ A^T
    // Call: cublasSgemm(OP_N, OP_N, n, m, k, B, n, A, k, C, n)
    // Leading dimensions based on number of columns in row-major storage
    
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        d_b, n,     // lda = n (cols in row-major B[k×n])
        d_a, k,     // ldb = k (cols in row-major A[m×k])
        &beta,
        out, n      // ldc = n (cols in row-major C[m×n])
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuda_matmul: cublasSgemm failed with status %d\n", status);
        fflush(stdout);
        return -1;
    }
    return 0;
}