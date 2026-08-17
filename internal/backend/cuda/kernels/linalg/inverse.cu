#include "ops_linalg.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// Pivot magnitude below which the matrix is declared singular.
#define INV_PIVOT_EPS 1e-12f

// ---------------------------------------------------------------------------
// Small path: Gauss-Jordan on [A | I] entirely in shared memory, one block.
//
// For small n the augmented matrix fits in shared memory, so the whole
// elimination runs without a single round trip to global memory. That matters
// more than the asymptotics here: at n = 8 a cuBLAS batched LU spends most of
// its time on launch and pointer setup rather than arithmetic.
//
// One thread per column of the augmented matrix (2n threads, <= 64). Each
// thread owns a column and walks it down the rows.
// ---------------------------------------------------------------------------
__global__ void inverse_gj_kernel(const float *__restrict__ A,
                                  float *__restrict__ out,
                                  int n, int *info) {
    extern __shared__ float aug[]; // n rows x 2n cols, row-major
    const int W = 2 * n;           // augmented row width
    const int j = threadIdx.x;     // this thread's column

    __shared__ int pivotRow;
    __shared__ float pivotVal;
    __shared__ int singular;
    __shared__ float fac[CUDA_INVERSE_SMALL_MAX];

    if (j == 0) singular = 0;
    __syncthreads();

    // Load [A | I].
    for (int i = 0; i < n; ++i) {
        aug[i * W + j] = (j < n) ? A[i * n + j]
                                 : (((j - n) == i) ? 1.0f : 0.0f);
    }
    __syncthreads();

    for (int k = 0; k < n; ++k) {
        // --- partial pivot: largest |value| at or below the diagonal.
        if (j == 0) {
            int p = k;
            float best = fabsf(aug[k * W + k]);
            for (int i = k + 1; i < n; ++i) {
                float v = fabsf(aug[i * W + k]);
                if (v > best) { best = v; p = i; }
            }
            pivotRow = p;
            pivotVal = aug[p * W + k];
            if (fabsf(pivotVal) < INV_PIVOT_EPS) singular = 1;
        }
        __syncthreads();

        // `singular` is uniform across the block, so this break cannot strand a
        // thread at a later __syncthreads().
        if (singular) break;

        // --- swap rows k and pivotRow.
        if (pivotRow != k) {
            float tmp = aug[k * W + j];
            aug[k * W + j] = aug[pivotRow * W + j];
            aug[pivotRow * W + j] = tmp;
        }
        __syncthreads();

        // --- normalise the pivot row. pivotVal was captured before any thread
        // writes to column k, so dividing by it is not a read-after-write race.
        const float pv = pivotVal;
        aug[k * W + j] /= pv;
        __syncthreads();

        // --- capture elimination factors BEFORE column k is overwritten.
        if (j < n) fac[j] = aug[j * W + k];
        __syncthreads();

        const float rowk = aug[k * W + j]; // row k is skipped below (i != k)
        for (int i = 0; i < n; ++i) {
            if (i != k) aug[i * W + j] -= fac[i] * rowk;
        }
        __syncthreads();
    }

    if (j == 0) *info = singular;
    if (singular) return;

    // The right half is the inverse.
    if (j < n) {
        for (int i = 0; i < n; ++i) out[i * n + j] = aug[i * W + (j + n)];
    }
}

int cuda_inverse_small(const float *d_A, float *d_out, int n, cudaStream_t stream) {
    if (!d_A || !d_out || n <= 0) {
        printf("cuda_inverse_small: invalid arguments (n=%d)\n", n);
        return -1;
    }
    if (n > CUDA_INVERSE_SMALL_MAX) {
        printf("cuda_inverse_small: n=%d exceeds %d\n", n, CUDA_INVERSE_SMALL_MAX);
        return -1;
    }

    int *d_info = NULL;
    if (cudaMalloc((void **)&d_info, sizeof(int)) != cudaSuccess) {
        printf("cuda_inverse_small: cudaMalloc(info) failed\n");
        return -1;
    }
    cudaMemsetAsync(d_info, 0, sizeof(int), stream);

    const size_t shmem = (size_t)n * (2 * n) * sizeof(float);
    inverse_gj_kernel<<<1, 2 * n, shmem, stream>>>(d_A, d_out, n, d_info);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cuda_inverse_small: launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_info);
        return -1;
    }

    int h_info = 0;
    cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    err = cudaStreamSynchronize(stream);
    cudaFree(d_info);

    if (err != cudaSuccess) {
        printf("cuda_inverse_small: execution failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return h_info ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Singularity check for the LU path.
//
// cublasSgetrfBatched only reports info > 0 for an *exactly* zero pivot. In
// float32 a rank-deficient matrix usually produces a tiny-but-nonzero pivot
// instead, so cuBLAS reports success and getri then returns garbage. Scanning
// the U diagonal for a pivot that is negligible relative to the largest one
// gives the LU path the same detection the Gauss-Jordan path already has.
//
// This is a cheap proxy for a condition estimate, not a substitute for one: it
// catches rank deficiency, not general ill-conditioning.
// ---------------------------------------------------------------------------
#define INV_RELATIVE_PIVOT_EPS 1e-6f

__global__ void lu_singular_check_kernel(const float *__restrict__ lu, int n, int *info) {
    // One block, one pass. n is small enough that a serial scan by thread 0 is
    // not worth parallelising against the LU that just ran.
    if (threadIdx.x != 0) return;

    float maxAbs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float v = fabsf(lu[i * n + i]);
        if (v > maxAbs) maxAbs = v;
    }
    if (maxAbs == 0.0f) { *info = 1; return; }

    const float tol = INV_RELATIVE_PIVOT_EPS * maxAbs;
    for (int i = 0; i < n; ++i) {
        if (fabsf(lu[i * n + i]) <= tol) { *info = 1; return; }
    }
}

// ---------------------------------------------------------------------------
// Large path: cuBLAS batched LU with batch size 1.
//
// cuBLAS is column-major while our buffers are row-major, but no transpose is
// needed: read column-major the buffer *is* A^T, and inv(A^T) = inv(A)^T.
// Written back column-major that is exactly inv(A) in row-major order.
//
// getrfBatched overwrites its input, so A is copied into scratch first -- the
// contract says d_A is preserved.
// ---------------------------------------------------------------------------
int cuda_inverse_large(const float *d_A, float *d_out, int n,
                       cublasHandle_t handle, cudaStream_t stream) {
    if (!handle || !d_A || !d_out || n <= 0) {
        printf("cuda_inverse_large: invalid arguments (n=%d)\n", n);
        return -1;
    }

    const size_t bytes = (size_t)n * n * sizeof(float);
    float *d_lu = NULL;
    float **d_Aarr = NULL;
    float **d_Carr = NULL;
    int *d_piv = NULL;
    int *d_info = NULL;
    int rc = -1;
    int h_info = 0;
    cublasStatus_t st;

    if (cudaMalloc((void **)&d_lu, bytes) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void **)&d_Aarr, sizeof(float *)) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void **)&d_Carr, sizeof(float *)) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void **)&d_piv, n * sizeof(int)) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void **)&d_info, sizeof(int)) != cudaSuccess) goto cleanup;

    if (cudaMemcpyAsync(d_lu, d_A, bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
        goto cleanup;

    // The batched API wants device arrays of device pointers, even for one item.
    if (cudaMemcpyAsync(d_Aarr, &d_lu, sizeof(float *), cudaMemcpyHostToDevice, stream) != cudaSuccess)
        goto cleanup;
    if (cudaMemcpyAsync(d_Carr, &d_out, sizeof(float *), cudaMemcpyHostToDevice, stream) != cudaSuccess)
        goto cleanup;

    cublasSetStream(handle, stream);

    st = cublasSgetrfBatched(handle, n, d_Aarr, n, d_piv, d_info, 1);
    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("cuda_inverse_large: getrfBatched failed (%d)\n", (int)st);
        goto cleanup;
    }

    if (cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        goto cleanup;
    if (cudaStreamSynchronize(stream) != cudaSuccess) goto cleanup;

    // info > 0 means U[info,info] is exactly zero: singular.
    if (h_info > 0) { rc = 1; goto cleanup; }
    if (h_info < 0) {
        printf("cuda_inverse_large: getrf bad argument %d\n", h_info);
        goto cleanup;
    }

    // cuBLAS only flags exact zeros, so also reject a negligible pivot.
    cudaMemsetAsync(d_info, 0, sizeof(int), stream);
    lu_singular_check_kernel<<<1, 1, 0, stream>>>(d_lu, n, d_info);
    if (cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        goto cleanup;
    if (cudaStreamSynchronize(stream) != cudaSuccess) goto cleanup;
    if (h_info != 0) { rc = 1; goto cleanup; }

    st = cublasSgetriBatched(handle, n, (const float *const *)d_Aarr, n,
                             d_piv, d_Carr, n, d_info, 1);
    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("cuda_inverse_large: getriBatched failed (%d)\n", (int)st);
        goto cleanup;
    }

    if (cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        goto cleanup;
    if (cudaStreamSynchronize(stream) != cudaSuccess) goto cleanup;

    rc = (h_info != 0) ? 1 : 0;

cleanup:
    if (d_lu) cudaFree(d_lu);
    if (d_Aarr) cudaFree(d_Aarr);
    if (d_Carr) cudaFree(d_Carr);
    if (d_piv) cudaFree(d_piv);
    if (d_info) cudaFree(d_info);
    return rc;
}

int cuda_inverse(const float *d_A, float *d_out, int n,
                 cublasHandle_t handle, cudaStream_t stream) {
    if (n <= 0) {
        printf("cuda_inverse: invalid n=%d\n", n);
        return -1;
    }
    if (n <= CUDA_INVERSE_SMALL_MAX) return cuda_inverse_small(d_A, d_out, n, stream);
    return cuda_inverse_large(d_A, d_out, n, handle, stream);
}
