#include <cuda_runtime.h>
#include "ops_dmas.h"
#include <stdio.h>

// -------------------------------------------------------------------------
// Kernel 1: Vectorized (The Workhorse)
// Processes 4 floats (128 bits) per thread to maximize memory bandwidth.
// -------------------------------------------------------------------------
__global__ void mul_vec4_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N_vec)
{
    // Cast to float4 to load 128 bits at a time
    // Note: cudaMalloc pointers are always 256-byte aligned, so this is safe.
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* out4 = reinterpret_cast<float4*>(out);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop ensures robustness for any size N_vec
    for (int i = idx; i < N_vec; i += stride) {
        float4 val_a = a4[i];
        float4 val_b = b4[i];
        float4 res;

        res.x = val_a.x * val_b.x;
        res.y = val_a.y * val_b.y;
        res.z = val_a.z * val_b.z;
        res.w = val_a.w * val_b.w;

        out4[i] = res;
    }
}

// -------------------------------------------------------------------------
// Kernel 2: Scalar (The Cleanup)
// Handles the remaining 1-3 elements if size is not divisible by 4.
// -------------------------------------------------------------------------
__global__ void mul_scalar_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int size,
    int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

int cuda_mul(float *d_a, float *d_b, float *out, int size, cudaStream_t stream) {
    
    const int blockSize = 256;
    
    // 2. Determine Vectorized vs Scalar split
    int vec_count = size / 4;
    int remainder = size % 4;

    // 3. Launch Vectorized Kernel
    if (vec_count > 0) {
        int gridSize = (vec_count + blockSize - 1) / blockSize;
        // Cap grid size to prevent excessive launch overhead on huge arrays
        if (gridSize > 65535) gridSize = 65535; 

        mul_vec4_kernel<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, out, vec_count);
    }

    // 4. Launch Scalar Kernel (only if size is not a multiple of 4)
    if (remainder > 0) {
        // Offset is where the vectorized part stopped
        int offset = vec_count * 4;
        mul_scalar_kernel<<<1, 32, 0, stream>>>(d_a, d_b, out, size, offset);
    }

    // 5. Check for launch errors
    // Note: This only catches launch errors, not async execution errors.
    // For async errors, you would need cudaStreamSynchronize, but that defeats the purpose of async.
    cudaError_t err = cudaGetLastError();
    
    // Return 0 for Success, 1 for Failure (Standard C/Go convention)
    return (err == cudaSuccess) ? 0 : 1;
}
