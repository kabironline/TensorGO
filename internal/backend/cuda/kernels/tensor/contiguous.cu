#include <cuda_runtime.h>
#include "ops_tensor.h"

// Kernel: map each linear output index to a physical index in src using
// shape and strides, then copy to dst.
__global__ void contiguous_kernel(const float *src, float *dst, int ndim, const int *shape, const int *strides, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    int physical = 0;
    int idx = i;
    for (int d = ndim - 1; d >= 0; --d) {
        int coord = idx % shape[d];
        physical += coord * strides[d];
        idx /= shape[d];
    }
    dst[i] = src[physical];
}

int cuda_contiguous(float *d_src, float *d_dst, int *h_shape, int *h_strides, int ndim, int offset, int total, cudaStream_t stream) {
    if (total <= 0 || ndim <= 0) return 0;

    // Copy shape & strides to device temporary buffers
    int *d_shape = nullptr;
    int *d_strides = nullptr;
    size_t nbytes = ndim * sizeof(int);
    cudaError_t err;

    err = cudaMalloc((void**)&d_shape, nbytes);
    if (err != cudaSuccess) return -1;
    err = cudaMemcpyAsync(d_shape, h_shape, nbytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cudaFree(d_shape); return -2; }

    err = cudaMalloc((void**)&d_strides, nbytes);
    if (err != cudaSuccess) { cudaFree(d_shape); return -3; }
    err = cudaMemcpyAsync(d_strides, h_strides, nbytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cudaFree(d_shape); cudaFree(d_strides); return -4; }

    // Launch kernel
    const int block = 256;
    int grid = (total + block - 1) / block;
    contiguous_kernel<<<grid, block, 0, stream>>>(d_src, d_dst, ndim, d_shape, d_strides, total);

    // Check kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_shape);
        cudaFree(d_strides);
        return -5;
    }

    // Free temporaries (safe after launch)
    cudaFree(d_shape);
    cudaFree(d_strides);

    return 0;
}
