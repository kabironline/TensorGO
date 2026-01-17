#include "ops_dmas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Adds two buffers element-wise: out = a + b
// Implementation: async copy b -> out on provided stream, set cuBLAS stream and SAXPY a into out
int cuda_add(float *d_a, float *d_b, float *out, int size, cudaStream_t stream, cublasHandle_t handle) {
    if (!d_a || !d_b || !out || size <= 0 || !handle) {
        return -1;
    }

    const float alpha = 1.0f;

    // Async copy d_b into out on the given stream
    cudaError_t err = cudaMemcpyAsync(out, d_b, size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
        return -1;
    }

    // Ensure cuBLAS uses the same stream
    cublasStatus_t status = cublasSetStream(handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }

    // Perform out = out + alpha * a (i.e., out <- out + a)
    status = cublasSaxpy_v2(handle, size, &alpha, d_a, 1, out, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }

    // Note: do not synchronize here; caller may manage the stream
    return 0;
}
