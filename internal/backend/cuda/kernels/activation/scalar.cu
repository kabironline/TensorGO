#include "ops_scalar.h"
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;

// ============================================================================
// Vectorized Kernels (4 floats per thread for better memory throughput)
// ============================================================================

__global__ void add_scalar_vec4_kernel(
    const float4* __restrict__ input,
    float scalar,
    float4* __restrict__ output,
    int n_vec4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n_vec4; i += stride) {
        float4 val = input[i];
        val.x += scalar;
        val.y += scalar;
        val.z += scalar;
        val.w += scalar;
        output[i] = val;
    }
}

__global__ void sub_scalar_vec4_kernel(
    const float4* __restrict__ input,
    float scalar,
    float4* __restrict__ output,
    int n_vec4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n_vec4; i += stride) {
        float4 val = input[i];
        val.x -= scalar;
        val.y -= scalar;
        val.z -= scalar;
        val.w -= scalar;
        output[i] = val;
    }
}

__global__ void mul_scalar_vec4_kernel(
    const float4* __restrict__ input,
    float scalar,
    float4* __restrict__ output,
    int n_vec4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n_vec4; i += stride) {
        float4 val = input[i];
        val.x *= scalar;
        val.y *= scalar;
        val.z *= scalar;
        val.w *= scalar;
        output[i] = val;
    }
}

__global__ void div_scalar_vec4_kernel(
    const float4* __restrict__ input,
    float scalar,
    float4* __restrict__ output,
    int n_vec4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float inv_scalar = 1.0f / scalar;  // Multiply by inverse is faster
    for (int i = idx; i < n_vec4; i += stride) {
        float4 val = input[i];
        val.x *= inv_scalar;
        val.y *= inv_scalar;
        val.z *= inv_scalar;
        val.w *= inv_scalar;
        output[i] = val;
    }
}

// ============================================================================
// Scalar Kernels (for remaining elements)
// ============================================================================

__global__ void add_scalar_kernel(
    const float* __restrict__ input,
    float scalar,
    float* __restrict__ output,
    int size,
    int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx + offset; i < size; i += stride) {
        output[i] = input[i] + scalar;
    }
}

__global__ void sub_scalar_kernel(
    const float* __restrict__ input,
    float scalar,
    float* __restrict__ output,
    int size,
    int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx + offset; i < size; i += stride) {
        output[i] = input[i] - scalar;
    }
}

__global__ void mul_scalar_kernel(
    const float* __restrict__ input,
    float scalar,
    float* __restrict__ output,
    int size,
    int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx + offset; i < size; i += stride) {
        output[i] = input[i] * scalar;
    }
}

__global__ void div_scalar_kernel(
    const float* __restrict__ input,
    float scalar,
    float* __restrict__ output,
    int size,
    int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float inv_scalar = 1.0f / scalar;
    for (int i = idx + offset; i < size; i += stride) {
        output[i] = input[i] * inv_scalar;
    }
}

// ============================================================================
// Host Functions
// ============================================================================

int cuda_add_scalar(float* input, float scalar, float* output, int size, cudaStream_t stream) {
    const int vec_size = 4;
    int n_vec = size / vec_size;
    int n_remain = size % vec_size;

    int grid_size_vec = (n_vec + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_size_scalar = (n_remain + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (n_vec > 0) {
        add_scalar_vec4_kernel<<<grid_size_vec, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const float4*>(input),
            scalar,
            reinterpret_cast<float4*>(output),
            n_vec
        );
    }

    if (n_remain > 0) {
        add_scalar_kernel<<<grid_size_scalar, BLOCK_SIZE, 0, stream>>>(
            input, scalar, output, size, n_vec * vec_size
        );
    }

    return 0;
}

int cuda_sub_scalar(float* input, float scalar, float* output, int size, cudaStream_t stream) {
    const int vec_size = 4;
    int n_vec = size / vec_size;
    int n_remain = size % vec_size;

    int grid_size_vec = (n_vec + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_size_scalar = (n_remain + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (n_vec > 0) {
        sub_scalar_vec4_kernel<<<grid_size_vec, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const float4*>(input),
            scalar,
            reinterpret_cast<float4*>(output),
            n_vec
        );
    }

    if (n_remain > 0) {
        sub_scalar_kernel<<<grid_size_scalar, BLOCK_SIZE, 0, stream>>>(
            input, scalar, output, size, n_vec * vec_size
        );
    }

    return 0;
}

int cuda_mul_scalar(float* input, float scalar, float* output, int size, cudaStream_t stream) {
    const int vec_size = 4;
    int n_vec = size / vec_size;
    int n_remain = size % vec_size;

    int grid_size_vec = (n_vec + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_size_scalar = (n_remain + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (n_vec > 0) {
        mul_scalar_vec4_kernel<<<grid_size_vec, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const float4*>(input),
            scalar,
            reinterpret_cast<float4*>(output),
            n_vec
        );
    }

    if (n_remain > 0) {
        mul_scalar_kernel<<<grid_size_scalar, BLOCK_SIZE, 0, stream>>>(
            input, scalar, output, size, n_vec * vec_size
        );
    }

    return 0;
}

int cuda_div_scalar(float* input, float scalar, float* output, int size, cudaStream_t stream) {
    const int vec_size = 4;
    int n_vec = size / vec_size;
    int n_remain = size % vec_size;

    int grid_size_vec = (n_vec + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_size_scalar = (n_remain + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (n_vec > 0) {
        div_scalar_vec4_kernel<<<grid_size_vec, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const float4*>(input),
            scalar,
            reinterpret_cast<float4*>(output),
            n_vec
        );
    }

    if (n_remain > 0) {
        div_scalar_kernel<<<grid_size_scalar, BLOCK_SIZE, 0, stream>>>(
            input, scalar, output, size, n_vec * vec_size
        );
    }

    return 0;
}
