#ifndef OPS_CONV_H
#define OPS_CONV_H

#include <cuda_runtime.h>
#include <cudnn.h>

#ifdef __cplusplus
extern "C" {
#endif

int cuda_convolution_forward(
	const float* input, const float* weights, const float* bias,
	float* output, int batch_size, int in_channels, int in_height, int in_width,
	int out_channels, int kernel_height, int kernel_width,
	int pad_height, int pad_width,
	int stride_height, int stride_width,
	cudaStream_t stream, cudnnHandle_t cudnn);

int cuda_convolution_backward(
	const float* input, const float* weights, const float* output_grad,
	float* input_grad, float* weights_grad, float* bias_grad,
	int batch_size, int in_channels, int in_height, int in_width,
	int out_channels, int kernel_height, int kernel_width,
	int pad_height, int pad_width,
	int stride_height, int stride_width,
	cudaStream_t stream, cudnnHandle_t cudnn);

#ifdef __cplusplus
}
#endif

#endif // OPS_CONV_H