#ifndef OPS_LINALG_H
#define OPS_LINALG_H

#include <cublas_v2.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Largest n handled by the shared-memory Gauss-Jordan path. Above this the
// augmented matrix (n x 2n floats) no longer fits comfortably in one block's
// shared memory, and cuBLAS's blocked LU wins anyway.
#define CUDA_INVERSE_SMALL_MAX 32

// cuda_inverse writes the inverse of the n x n ROW-MAJOR matrix d_A into d_out.
// d_A is not modified. d_A and d_out must not alias.
//
// Dispatches on n: a single-block Gauss-Jordan kernel for small matrices, and
// cuBLAS batched LU (getrf + getri) for large ones.
//
// Returns 0 on success, 1 if the matrix is singular, and -1 on a bad argument
// or a CUDA/cuBLAS failure.
int cuda_inverse(const float *d_A, float *d_out, int n,
                 cublasHandle_t handle, cudaStream_t stream);

// Force a specific path regardless of n. For tests and benchmarks only -- the
// small path fails with -1 when n > CUDA_INVERSE_SMALL_MAX.
int cuda_inverse_small(const float *d_A, float *d_out, int n, cudaStream_t stream);
int cuda_inverse_large(const float *d_A, float *d_out, int n,
                       cublasHandle_t handle, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
