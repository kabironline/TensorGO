#include "ops_optim.h"
#include <cuda_runtime.h>
#include <math.h>

static int blockSize = 256;

// ============================================================================
// SGD Kernel
// ============================================================================
__global__ void sgd_kernel(float *data, const float *grad, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        data[i] -= lr * grad[i];
    }
}

int cuda_step_sgd(float *data, const float *grad, float lr, int size, cudaStream_t stream) {
    if (size <= 0) return 0;

    int gridSize = (size + blockSize - 1) / blockSize;
    sgd_kernel<<<gridSize, blockSize, 0, stream>>>(data, grad, lr, size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}
