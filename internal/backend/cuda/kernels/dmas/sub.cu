#include "ops_dmas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

// -------------------------------------------------------------------------
// Kernel 1: Vectorized Sub (The Workhorse)
// Processes 4 floats (128 bits) per thread to maximize memory bandwidth.
// -------------------------------------------------------------------------
__global__ void sub_vec4_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N_vec)
{
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* out4 = reinterpret_cast<float4*>(out);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N_vec; i += stride) {
        float4 val_a = a4[i];
        float4 val_b = b4[i];
        float4 res;

        res.x = val_a.x - val_b.x;
        res.y = val_a.y - val_b.y;
        res.z = val_a.z - val_b.z;
        res.w = val_a.w - val_b.w;

        out4[i] = res;
    }
}

// -------------------------------------------------------------------------
// Kernel 2: Scalar Sub (The Cleanup)
// Handles the remaining 1-3 elements if size is not divisible by 4.
// -------------------------------------------------------------------------
__global__ void sub_scalar_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int size,
    int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

// Subtract two buffers element-wise: out = a - b
// Uses optimized vectorized CUDA kernels
int cuda_sub(float *d_a, float *d_b, float *out, int size, cudaStream_t stream, cublasHandle_t handle) {
    if (!d_a || !d_b || !out || size <= 0) {
        printf("cuda_sub: Invalid parameters\n");
        return -1;
    }

    // Clear any previous errors
    cudaGetLastError();

    const int blockSize = 256;
    
    // Determine Vectorized vs Scalar split
    int vec_count = size / 4;
    int remainder = size % 4;

    // Launch Vectorized Kernel
    if (vec_count > 0) {
        int gridSize = (vec_count + blockSize - 1) / blockSize;
        if (gridSize > 65535) gridSize = 65535; 

        sub_vec4_kernel<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, out, vec_count);
    }

    // Launch Scalar Kernel (only if size is not a multiple of 4)
    if (remainder > 0) {
        int offset = vec_count * 4;
        sub_scalar_kernel<<<1, 32, 0, stream>>>(d_a, d_b, out, size, offset);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cuda_sub: Kernel launch failed with error %d: %s\n", err, cudaGetErrorString(err));
        fflush(stdout);
        return 1;
    }
    
    // Synchronize to catch execution errors
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        printf("cuda_sub: Kernel execution failed with error %d: %s\n", err, cudaGetErrorString(err));
        fflush(stdout);
        return 1;
    }
    return 0;
}
