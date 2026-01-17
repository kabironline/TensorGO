#include "matrix_ops.h"
#include <cublas_v2.h>

int cuda_matmul(float *d_a, float *d_b, float *out, int m, int n, int k, int sA, int sB, cublasHandle_t handle) {
     if (!handle || !d_a || !d_b || !out) {
        return -1;
    }

    if (m <= 0 || n <= 0 || k <= 0) {
        return -1;
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    
    // cuBLAS expects column-major, we have row-major
    // To compute C = A @ B in row-major:
    // We compute C^T = B^T @ A^T in column-major
    // This means we swap A and B, and swap m and n
    
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N,           // B is not transposed (from cuBLAS perspective)
        CUBLAS_OP_N,           // A is not transposed (from cuBLAS perspective)
        n,                     // Rows of B^T (= cols of B)
        m,                     // Cols of A^T (= rows of A)
        k,                     // Inner dimension
        &alpha,
        d_b, sB,         // B comes first
        d_a, sA,         // A comes second
        &beta,
        out, n               // Leading dimension of output
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }
    return 0;
}