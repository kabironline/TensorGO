#include "ops_activation.h"
#include <cuda_runtime.h>


int blockSize = 256;

// -------------------------------------------------------------------------
// Kernel 1: Vectorized (The Workhorse)
// Processes 4 floats (128 bits) per thread to maximize memory bandwidth.
// -------------------------------------------------------------------------
__global__ void relu_vec4_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N_vec)
{
    // Cast to float4 to load 128 bits at a time
    // Note: cudaMalloc pointers are always 256-byte aligned, so this is safe.
    const float4* in4 = reinterpret_cast<const float4*>(input);
    float4* out4 = reinterpret_cast<float4*>(output);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stide = blockDim.x * gridDim.x;

    for (int i = idx; i < N_vec; i += stide) {
        float4 val = in4[i];
        // Apply ReLU
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        out4[i] = val;
    }
}

// -------------------------------------------------------------------------
// Kernel 2: Scalar (Handles Remaining Elements)
// Processes one float per thread for any leftover elements.
// -------------------------------------------------------------------------
__global__ void relu_scalar_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int offset) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stide = blockDim.x * gridDim.x;
    for (int i = idx + offset; i < N; i += stide) {
        float val = input[i];
        // Apply ReLU
        output[i] = fmaxf(val, 0.0f);
    }
}


int cuda_relu(
    float* d_in,
    float* out,
    int size,
    cudaStream_t stream)
{
    const int vec_size = 4; // Number of floats processed per thread in vectorized kernel
    int N_vec = size / vec_size; // Number of full vectorized elements
    int N_remain = size % vec_size; // Remaining elements for scalar kernel

    // Launch parameters
    int gridSize_vec = (N_vec + blockSize - 1) / blockSize;
    int gridSize_scalar = (N_remain + blockSize - 1) / blockSize;

    // Launch vectorized kernel
    if (N_vec > 0) {
        relu_vec4_kernel<<<gridSize_vec, blockSize, 0, stream>>>(d_in, out, N_vec);
    }

    // Launch scalar kernel for remaining elements
    if (N_remain > 0) {
        relu_scalar_kernel<<<gridSize_scalar, blockSize, 0, stream>>>(d_in, out, size, N_vec * vec_size);
    }

    return 0;
}


// ---------------------------------------------------------------------------------
// Kernel for Vectorized RELU Backwards
// --------------------------------------------------------------------------------
__global__ void relu_vec4_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int N_vec)
{
    const float4* grad_out4 = reinterpret_cast<const float4*>(grad_output);
    const float4* in4 = reinterpret_cast<const float4*>(input);
    float4* grad_in4 = reinterpret_cast<float4*>(grad_input);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stide = blockDim.x * gridDim.x;

    for (int i = idx; i < N_vec; i += stide) {
        float4 grad_out = grad_out4[i];
        float4 in_val = in4[i];
        float4 grad_in;

        // Apply ReLU backward
        grad_in.x = (in_val.x > 0.0f) ? grad_out.x : 0.0f;
        grad_in.y = (in_val.y > 0.0f) ? grad_out.y : 0.0f;
        grad_in.z = (in_val.z > 0.0f) ? grad_out.z : 0.0f;
        grad_in.w = (in_val.w > 0.0f) ? grad_out.w : 0.0f;

        grad_in4[i] = grad_in;
    }
}

// ---------------------------------------------------------------------------------
// Kernel for Scalar RELU Backwards
// ---------------------------------------------------------------------------------

__global__ void relu_scalar_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int N,
    int offset) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stide = blockDim.x * gridDim.x;
    for (int i = idx + offset; i < N; i += stide) {
        float grad_out = grad_output[i];
        float in_val = input[i];
        // Apply ReLU backward
        grad_input[i] = (in_val > 0.0f) ? grad_out : 0.0f;
    }
}

int cuda_relu_backward(
    float* d_in,
    float* d_in_grad,
    float* out,
    int size,
    cudaStream_t stream)
{
    const int vec_size = 4; // Number of floats processed per thread in vectorized kernel
    int N_vec = size / vec_size; // Number of full vectorized elements
    int N_remain = size % vec_size; // Remaining elements for scalar kernel

    // Launch parameters
    int gridSize_vec = (N_vec + blockSize - 1) / blockSize;
    int gridSize_scalar = (N_remain + blockSize - 1) / blockSize;

    // Launch vectorized backward kernel
    if (N_vec > 0) {
        relu_vec4_backward_kernel<<<gridSize_vec, blockSize, 0, stream>>>(d_in_grad, d_in, out, N_vec);
    }

    // Launch scalar backward kernel for remaining elements
    if (N_remain > 0) {
        relu_scalar_backward_kernel<<<gridSize_scalar, blockSize, 0, stream>>>(d_in_grad, d_in, out, size, N_vec * vec_size);
    }

    return 0;
}