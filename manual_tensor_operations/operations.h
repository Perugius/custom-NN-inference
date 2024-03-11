#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <vector>

// Declare single_filter_1Dconvolution function
std::vector<float> single_filter_1Dconvolution(const std::vector<float>& input, const std::vector<std::vector<float>>& kernel, int channel_length, int num_of_channels, int stride);

// Declare multi_filter_1Dconvolution function
std::vector<std::vector<float>> multi_filter_1Dconvolution(const std::vector<float>& input, const std::vector<std::vector<std::vector<float>>>& filters, const std::vector<float>& conv_bias, int channel_length, int num_of_channels, int stride, int num_of_filters);

// Declare flatten function
std::vector<float> flatten(std::vector<std::vector<float>>& input);

// Declare fully_connected function
std::vector<float> fully_connected(const std::vector<float>& input, const std::vector<std::vector<float>>& weights, const std::vector<float>& fully_connected_bias);

#endif // OPERATIONS_H