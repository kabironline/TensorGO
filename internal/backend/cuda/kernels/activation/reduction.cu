#include "ops_reduction.h"
#include <cuda_runtime.h>
#include <stdio.h>

static const int BLOCK_SIZE = 256;
static const int MAX_BLOCKS = 1024;

// ============================================================================
// Block-level Sum Reduction using Shared Memory
// ============================================================================

__global__ void sum_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size)
{
    __shared__ float shared[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread accumulates multiple elements
    float sum = 0.0f;
    for (int i = idx; i < size; i += stride) {
        sum += input[i];
    }
    shared[tid] = sum;
    __syncthreads();

    // Tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

// ============================================================================
// Host Function: Sum
// ============================================================================

float cuda_sum(float* input, int size, cudaStream_t stream) {
    if (size == 0) return 0.0f;

    // Allocate device memory for partial sums
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > MAX_BLOCKS) {
        num_blocks = MAX_BLOCKS;
    }
    float* d_partial_sums;
    cudaMalloc(&d_partial_sums, num_blocks * sizeof(float));

    // First reduction: input -> partial sums (one per block)
    sum_reduction_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, d_partial_sums, size);

    // If we have multiple blocks, reduce them
    if (num_blocks > 1) {
        // Allocate for second-level reduction
        float* d_final_sum;
        cudaMalloc(&d_final_sum, sizeof(float));

        // Second reduction: partial sums -> final sum
        sum_reduction_kernel<<<1, BLOCK_SIZE, 0, stream>>>(d_partial_sums, d_final_sum, num_blocks);

        // Copy result to host
        float result;
        cudaMemcpyAsync(&result, d_final_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cudaFree(d_final_sum);
        cudaFree(d_partial_sums);
        return result;
    } else {
        // Only one block, copy directly
        float result;
        cudaMemcpyAsync(&result, d_partial_sums, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaFree(d_partial_sums);
        return result;
    }
}

// ============================================================================
// Host Function: Mean
// ============================================================================

float cuda_mean(float* input, int size, cudaStream_t stream) {
    if (size == 0) return 0.0f;
    float sum = cuda_sum(input, size, stream);
    return sum / float(size);
}

// ============================================================================
// SumAxis Kernel - Reduces along a specific dimension
// ============================================================================

__global__ void sum_axis_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int outer_size,      // Product of dimensions before axis
    int axis_size,       // Size of the axis being reduced
    int inner_size)      // Product of dimensions after axis
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = outer_size * inner_size;

    if (idx < total_output) {
        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;

        float sum = 0.0f;
        for (int axis_idx = 0; axis_idx < axis_size; axis_idx++) {
            int input_idx = outer_idx * (axis_size * inner_size) +
                          axis_idx * inner_size +
                          inner_idx;
            sum += input[input_idx];
        }

        output[idx] = sum;
    }
}

// ============================================================================
// Host Function: SumAxis
// ============================================================================

void cuda_sum_axis(float* input, int* shape, int ndim, int axis, float* output, int out_size, cudaStream_t stream) {
    if (axis < 0 || axis >= ndim) {
        printf("Invalid axis %d for ndim %d\n", axis, ndim);
        return;
    }

    // Calculate dimensions
    int outer_size = 1;
    for (int i = 0; i < axis; i++) {
        outer_size *= shape[i];
    }

    int axis_size = shape[axis];

    int inner_size = 1;
    for (int i = axis + 1; i < ndim; i++) {
        inner_size *= shape[i];
    }

    int total_output = outer_size * inner_size;

    if (out_size != total_output) {
        printf("Invalid out_size %d (expected %d)\n", out_size, total_output);
        return;
    }

    // Launch kernel
    int num_blocks = (total_output + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sum_axis_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        input, output, outer_size, axis_size, inner_size
    );
}
