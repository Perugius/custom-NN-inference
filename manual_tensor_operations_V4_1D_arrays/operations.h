#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <vector>

// performs the depthwise convolution part of the separable convolution
float* depthwise_conv1d(float* input, const std::vector<std::vector<float>>& depthwise_weights, int stride, int& input_length, int& num_of_channels);

// performs the pointwise convolution part of the separable convolution
float* pointwise_conv1d(float* input, const std::vector<std::vector<float>>& pointwise_weights, int& input_length, int& num_of_channels);

void batch_normalization(float* input, const std::vector<float>& gamma_coeff, const std::vector<float>& beta, const std::vector<float>& moving_mean, const std::vector<float>& moving_variance, int input_length, int num_of_channels, float epsilon);

float* fully_connected(float* input, const std::vector<std::vector<float>>& weights, int& input_length, int& num_of_channels);

float* global_pooling1D(float* input, int& input_length, int& num_of_channels);

void relu(float* input, int input_length, int num_of_channels);

#endif // OPERATIONS_H