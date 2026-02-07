#include "ops_optim.h"
#include <cuda_runtime.h>
#include <math.h>

static int blockSize = 256;

// ============================================================================
// Adam Kernel - Vectorized (processes 4 elements per thread)
// ============================================================================
__global__ void adam_vec4_kernel(
    float *data,
    const float *grad,
    float *m,
    float *v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float beta1_pow,
    float beta2_pow,
    float one_minus_beta1,
    float one_minus_beta2,
    int size_vec)
{
    // Cast to float4 for vectorized operations
    float4 *data4 = reinterpret_cast<float4 *>(data);
    const float4 *grad4 = reinterpret_cast<const float4 *>(grad);
    float4 *m4 = reinterpret_cast<float4 *>(m);
    float4 *v4 = reinterpret_cast<float4 *>(v);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float denom1 = 1.0f - beta1_pow;
    float denom2 = 1.0f - beta2_pow;
    float lr_over_denom1 = lr / denom1;

    for (int i = idx; i < size_vec; i += stride) {
        float4 g = grad4[i];
        float4 m_val = m4[i];
        float4 v_val = v4[i];
        float4 data_val = data4[i];

        // Update m and v for each element
        m_val.x = beta1 * m_val.x + one_minus_beta1 * g.x;
        m_val.y = beta1 * m_val.y + one_minus_beta1 * g.y;
        m_val.z = beta1 * m_val.z + one_minus_beta1 * g.z;
        m_val.w = beta1 * m_val.w + one_minus_beta1 * g.w;

        v_val.x = beta2 * v_val.x + one_minus_beta2 * g.x * g.x;
        v_val.y = beta2 * v_val.y + one_minus_beta2 * g.y * g.y;
        v_val.z = beta2 * v_val.z + one_minus_beta2 * g.z * g.z;
        v_val.w = beta2 * v_val.w + one_minus_beta2 * g.w * g.w;

        // Bias correction and parameter update
        data_val.x -= lr_over_denom1 * m_val.x / (sqrtf(v_val.x / denom2) + eps);
        data_val.y -= lr_over_denom1 * m_val.y / (sqrtf(v_val.y / denom2) + eps);
        data_val.z -= lr_over_denom1 * m_val.z / (sqrtf(v_val.z / denom2) + eps);
        data_val.w -= lr_over_denom1 * m_val.w / (sqrtf(v_val.w / denom2) + eps);

        // Write back
        data4[i] = data_val;
        m4[i] = m_val;
        v4[i] = v_val;
    }
}

// ============================================================================
// Adam Kernel - Scalar (handles remaining elements)
// ============================================================================
__global__ void adam_scalar_kernel(
    float *data,
    const float *grad,
    float *m,
    float *v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float beta1_pow,
    float beta2_pow,
    float one_minus_beta1,
    float one_minus_beta2,
    int size,
    int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float denom1 = 1.0f - beta1_pow;
    float denom2 = 1.0f - beta2_pow;
    float lr_over_denom1 = lr / denom1;

    for (int i = idx + offset; i < size; i += stride) {
        float g = grad[i];
        float m_val = m[i];
        float v_val = v[i];

        // Update m and v
        m_val = beta1 * m_val + one_minus_beta1 * g;
        v_val = beta2 * v_val + one_minus_beta2 * g * g;

        // Bias correction and parameter update
        data[i] -= lr_over_denom1 * m_val / (sqrtf(v_val / denom2) + eps);

        // Write back
        m[i] = m_val;
        v[i] = v_val;
    }
}

int cuda_step_adam(float *data, const float *grad, float *m, float *v,
                   float lr, float beta1, float beta2, float eps, int t, int size, cudaStream_t stream)
{
    if (size <= 0 || t <= 0) return 0;

    // Pre-compute bias correction terms on host (cheaper than on device)
    float beta1_pow = powf(beta1, (float)t);
    float beta2_pow = powf(beta2, (float)t);
    float one_minus_beta1 = 1.0f - beta1;
    float one_minus_beta2 = 1.0f - beta2;

    const int vec_size = 4;
    int size_vec = size / vec_size;
    int size_remain = size % vec_size;

    int gridSize_vec = (size_vec + blockSize - 1) / blockSize;
    int gridSize_scalar = (size_remain + blockSize - 1) / blockSize;

    // Launch vectorized kernel
    if (size_vec > 0) {
        adam_vec4_kernel<<<gridSize_vec, blockSize, 0, stream>>>(
            data, grad, m, v, lr, beta1, beta2, eps,
            beta1_pow, beta2_pow, one_minus_beta1, one_minus_beta2, size_vec);
    }

    // Launch scalar kernel for remaining elements
    if (size_remain > 0) {
        adam_scalar_kernel<<<gridSize_scalar, blockSize, 0, stream>>>(
            data, grad, m, v, lr, beta1, beta2, eps,
            beta1_pow, beta2_pow, one_minus_beta1, one_minus_beta2, size, size_vec * vec_size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    return 0;
}
