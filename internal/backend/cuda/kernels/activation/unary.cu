#include <cuda_runtime.h>
#include <cmath>
#include "ops_unary.h"

static const int blockSize = 256;

// Exponential kernel
__global__ void exp_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = expf(input[idx]);
    }
}

void cuda_exp(const float* input, float* output, int size) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    exp_kernel<<<numBlocks, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
}

// Logarithm kernel
__global__ void log_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = logf(input[idx]);
    }
}

void cuda_log(const float* input, float* output, int size) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    log_kernel<<<numBlocks, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
}

// Square kernel
__global__ void square_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = val * val;
    }
}

void cuda_square(const float* input, float* output, int size) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    square_kernel<<<numBlocks, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
}

// Negation kernel
__global__ void neg_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = -input[idx];
    }
}

void cuda_neg(const float* input, float* output, int size) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    neg_kernel<<<numBlocks, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
}

// Square root kernel
__global__ void sqrt_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = sqrtf(input[idx]);
    }
}

void cuda_sqrt(const float* input, float* output, int size) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    sqrt_kernel<<<numBlocks, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
}
