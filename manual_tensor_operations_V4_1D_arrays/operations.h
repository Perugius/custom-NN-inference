#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <vector>

// Deallocate a dynamically allocated 2D array
template<typename T>
void deallocate_array(T**& array, int first_dim_size);

// Performs the depthwise convolution part of the separable convolution
float* depthwise_conv1d(float* input, const std::vector<std::vector<float>>& depthwise_weights, int stride, int& input_length, int& num_of_channels);

// Performs the pointwise convolution part of the separable convolution
float* pointwise_conv1d(float* input, const std::vector<std::vector<float>>& pointwise_weights, int& input_length, int& num_of_channels);

// Combine depthwise and pointwise convolution operations
float* separable_conv1d(float* input, const std::vector<std::vector<float>>& depthwise_weights, int stride, int& input_length, int& num_of_channels, const std::vector<std::vector<float>>& pointwise_weights);

#endif // OPERATIONS_H