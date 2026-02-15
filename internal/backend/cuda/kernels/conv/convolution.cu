#include "ops_conv.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>

#define CUDNN_CHECK(call) do { \
    cudnnStatus_t status = (call); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error: %s\n", cudnnGetErrorString(status)); \
        return -1; \
    } \
} while (0)

#define CUDA_CHECK(call) do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status)); \
        return -1; \
    } \
} while (0)

static void* g_workspace = nullptr;
static size_t g_workspace_size = 0;

static int ensure_workspace(size_t size, void** out_ptr) {
    if (size == 0) {
        *out_ptr = nullptr;
        return 0;
    }

    if (size > g_workspace_size) {
        if (g_workspace != nullptr) {
            cudaError_t err = cudaFree(g_workspace);
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
                return -1;
            }
            g_workspace = nullptr;
            g_workspace_size = 0;
        }
        cudaError_t err = cudaMalloc(&g_workspace, size);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        g_workspace_size = size;
    }

    *out_ptr = g_workspace;
    return 0;
}


// ============================================================================
// Convolution Forward Pass using cuDNN
// ============================================================================
extern "C" int cuda_convolution_forward(
    const float* input, const float* weights, const float* bias,
    float* output, int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_height, int kernel_width,
    int pad_height, int pad_width,
    int stride_height, int stride_width,
    cudaStream_t stream, cudnnHandle_t cudnn)
{
    CUDNN_CHECK(cudnnSetStream(cudnn, stream));

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set input descriptor
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, in_channels, in_height, in_width));

    // Set filter descriptor
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc,
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        out_channels, in_channels, kernel_height, kernel_width));

    // Set convolution descriptor
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
        pad_height, pad_width,
        stride_height, stride_width,
        1, 1, // dilation
        // cudnnConvolutionMode_t mode, cudnnDataType_t computeType
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Get output dimensions
    int out_n = 0, out_c = 0, out_height = 0, out_width = 0;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, input_desc, filter_desc, &out_n, &out_c, &out_height, &out_width));

    // Set output descriptor
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_height, out_width));

    // Allocate workspace for convolution
    int returned_algo_count = 0;
    cudnnConvolutionFwdAlgoPerf_t fwd_perf[8];
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn, input_desc, filter_desc, conv_desc, output_desc,
        8, &returned_algo_count, fwd_perf));
    if (returned_algo_count <= 0) {
        fprintf(stderr, "cuDNN error: no forward algorithms found\n");
        return -1;
    }

    size_t workspace_size = 0;
    void* workspace = nullptr;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        input_desc, filter_desc, conv_desc, output_desc,
        fwd_perf[0].algo,
        &workspace_size));
    if (ensure_workspace(workspace_size, &workspace) != 0) {
        return -1;
    }
    // Perform convolution
    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(cudnn,
        &alpha, input_desc, input,
        filter_desc, weights,
        conv_desc, fwd_perf[0].algo
        , workspace, workspace_size,
        &beta, output_desc, output));

    // Add bias: bias is [out_channels] and broadcast as [1, C, 1, 1].
    if (bias != nullptr) {
        cudnnTensorDescriptor_t bias_desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            1, out_c, 1, 1));
        const float bias_alpha = 1.0f;
        const float bias_beta = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(cudnn,
            &bias_alpha, bias_desc, bias,
            &bias_beta, output_desc, output));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    }

    // Clean up
    (void)workspace;
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    // cudnnDestroy(cudnn);
    return 0;
}

// ============================================================================
// Convolution Backward Pass using cuDNN
// ============================================================================
extern "C" int cuda_convolution_backward(
    const float* input, const float* weights, const float* output_grad,
    float* input_grad, float* weights_grad, float* bias_grad,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_height, int kernel_width,
    int pad_height, int pad_width,
    int stride_height, int stride_width,
    cudaStream_t stream, cudnnHandle_t cudnn)
{
    CUDNN_CHECK(cudnnSetStream(cudnn, stream));
    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc, output_grad_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_grad_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    // Set input descriptor
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, in_channels, in_height, in_width));
    // Set filter descriptor
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc,
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        out_channels, in_channels, kernel_height, kernel_width));
    // Set convolution descriptor
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
        pad_height, pad_width,
        stride_height, stride_width,
        1, 1, // dilation
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    // Get output dimensions
    int out_n = 0, out_c = 0, out_height = 0, out_width = 0;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, input_desc, filter_desc, &out_n, &out_c, &out_height, &out_width));
    // Set output descriptor
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_height, out_width));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_grad_desc,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_height, out_width));
    // Allocate workspace for convolution
    size_t workspace_size = 0;
    void* workspace = nullptr;
    int returned_bwd_data_count = 0;
    int returned_bwd_filter_count = 0;
    cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf[8];
    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perf[8];

    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        cudnn, filter_desc, output_grad_desc, conv_desc, input_desc,
        8, &returned_bwd_data_count, bwd_data_perf));
    if (returned_bwd_data_count <= 0) {
        fprintf(stderr, "cuDNN error: no backward-data algorithms found\n");
        return -1;
    }

    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        cudnn, input_desc, output_grad_desc, conv_desc, filter_desc,
        8, &returned_bwd_filter_count, bwd_filter_perf));
    if (returned_bwd_filter_count <= 0) {
        fprintf(stderr, "cuDNN error: no backward-filter algorithms found\n");
        return -1;
    }

    size_t workspace_size_data = 0;
    size_t workspace_size_filter = 0;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn,
        filter_desc, output_grad_desc, conv_desc, input_desc,
        bwd_data_perf[0].algo,
        &workspace_size_data));
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn,
        input_desc, output_grad_desc, conv_desc, filter_desc,
        bwd_filter_perf[0].algo,
        &workspace_size_filter));
    workspace_size = (workspace_size_data > workspace_size_filter)
        ? workspace_size_data
        : workspace_size_filter;
    if (ensure_workspace(workspace_size, &workspace) != 0) {
        return -1;
    }
    // Compute input gradient
    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardData(cudnn,
        &alpha, filter_desc, weights,
        output_grad_desc, output_grad,
        conv_desc, bwd_data_perf[0].algo,
        workspace, workspace_size,
        &beta, input_desc, input_grad));
    // Compute weights gradient
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(cudnn,
        &alpha, input_desc, input,
        output_grad_desc, output_grad,
        conv_desc, bwd_filter_perf[0].algo,
        workspace, workspace_size,
        &beta, filter_desc, weights_grad));
    // Compute bias gradient
    if (bias_grad != nullptr) {
        cudnnTensorDescriptor_t bias_desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            1, out_c, 1, 1));
        CUDNN_CHECK(cudnnConvolutionBackwardBias(cudnn,
            &alpha, output_grad_desc, output_grad,
            &beta, bias_desc, bias_grad));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    }
    // Clean up
    (void)workspace;
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_grad_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));

    return 0;
}