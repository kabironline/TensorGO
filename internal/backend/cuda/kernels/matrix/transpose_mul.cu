#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include "ops_matrix.h"

// Matrix multiplication with A transposed: C = A^T @ B
// Following CPU backend convention:
// A is (k x m) stored row-major
// B is (k x n) stored row-major
// C is (m x n) stored row-major
void cuda_matmul_trans_a(const float* A, const float* B, float* C,
                         int m, int n, int k, int strideA, int strideB, cublasHandle_t handle) {
    if (!handle || !A || !B || !C) return;
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Verified formula: sgemm(OP_N, OP_T, n, m, k, B, n, A, m, C, n)
    // lda=n (cols of B[k×n]), ldb=m (cols of A[k×m])
    
    cublasStatus_t status = cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                n, m, k,
                &alpha,
                B, n,       // lda = n (cols of B[k×n] in row-major)
                A, m,       // ldb = m (cols of A[k×m] in row-major)
                &beta,
                C, n);      // ldc = n (cols of C[m×n] in row-major)
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuda_matmul_trans_a: cublasSgemm failed with status %d (m=%d n=%d k=%d)\n", 
               status, m, n, k);
        fflush(stdout);
    }
    
    cudaDeviceSynchronize();
}

// Matrix multiplication with B transposed: C = A @ B^T
// Following CPU backend convention:
// A is (m x k) stored row-major
// B is (n x k) stored row-major, we use B^T which is (k x n)
// C is (m x n) stored row-major
void cuda_matmul_trans_b(const float* A, const float* B, float* C,
                         int m, int n, int k, int strideA, int strideB, cublasHandle_t handle) {
    if (!handle || !A || !B || !C) return;
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Verified formula: sgemm(OP_T, OP_N, n, m, k, B, k, A, k, C, n)
   // Works with row-major data - cuBLAS output is row-major compatible
    
    cublasStatus_t status = cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                B, k,       // lda = k (cols of B[n×k] in row-major)
                A, k,       // ldb = k (cols of A[m×k] in row-major)
                &beta,
                C, n);      // ldc = n (cols of C[m×n] in row-major)
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuda_matmul_trans_b: cublasSgemm failed with status %d (m=%d n=%d k=%d)\n", 
               status, m, n, k);
        fflush(stdout);
    }
    
    cudaDeviceSynchronize();
}

