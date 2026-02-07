#ifndef OPS_UNARY_H
#define OPS_UNARY_H

#ifdef __cplusplus
extern "C" {
#endif

void cuda_exp(const float* input, float* output, int size);
void cuda_log(const float* input, float* output, int size);
void cuda_square(const float* input, float* output, int size);
void cuda_neg(const float* input, float* output, int size);
void cuda_sqrt(const float* input, float* output, int size);

#ifdef __cplusplus
}
#endif

#endif // OPS_UNARY_H