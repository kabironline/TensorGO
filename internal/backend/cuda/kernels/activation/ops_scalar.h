#ifndef OPS_SCALAR_H
#define OPS_SCALAR_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

int cuda_add_scalar(float* input, float scalar, float* output, int size, cudaStream_t stream);
int cuda_sub_scalar(float* input, float scalar, float* output, int size, cudaStream_t stream);
int cuda_mul_scalar(float* input, float scalar, float* output, int size, cudaStream_t stream);
int cuda_div_scalar(float* input, float scalar, float* output, int size, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // OPS_SCALAR_H
