#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "operations.h"

//convolution of 1 filter only, result will be a 1d array
std::vector<float> single_filter_1Dconvolution(const std::vector<float>& input, const std::vector<std::vector<float>>& kernel, int channel_length, int num_of_channels, int stride) {

    int input_size = static_cast<int>(channel_length);
    int kernel_size = static_cast<int>(kernel[0].size());
    int result_size = static_cast<int>(std::floor(input_size - kernel_size)/stride) + 1;
    std::vector<float> result(result_size, 0.0);
    // take one of the input vectors and perform convolution and add it to the result
    std::vector<float> single_input(channel_length, 0.0);

    for (int k = 0; k < num_of_channels; ++k){
        // iterate over vectors because input is flat so a (80, 3) shape will be represented as (240) -> iterate over 3 vectors of length 80
        // each of those single vectors is saved into single_input
        std::copy(input.begin() + k*channel_length, input.begin() + k*channel_length + channel_length, single_input.begin());
        // i iterates over result (i.e kernel sliding over input vector), j iterates over the kernel itself multiplying and adding into the results
        // finally k iterates over all the kernels and adds up everything to the same result
        for (int i = 0; i < result_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                result[i] += single_input[i*stride + j] * kernel[k][j];
            }
        }
    }
    return result;
}

//uses single_filter_IDconvolution function to convolute over multiple filters, result is 2d vector of shape (filters, channel_length)
std::vector<std::vector<float>> multi_filter_1Dconvolution(const std::vector<float>& input, const std::vector<std::vector<std::vector<float>>>& filters, const std::vector<float>& conv_bias, int channel_length, int num_of_channels, int stride, int num_of_filters){
    std::vector<std::vector<float>> result_multi;
    //iterate over all filters passing the kernels to the 1dconv function

    for (int i = 0; i < num_of_filters; ++i){
        std::vector<float> single_result = single_filter_1Dconvolution(input, filters[i], channel_length, num_of_channels, stride);
        for (int j = 0; j < single_result.size(); ++j){
            single_result[j] += conv_bias[i];
        }
        result_multi.push_back(single_result);
    }

    return result_multi;
}

//Flatten function
std::vector<float> flatten(std::vector<std::vector<float>>& input){
    std::vector<float> output;

    for (const auto& values : input){
        output.insert(output.end(), values.begin(), values.end());
    }
    return output;
}

//dense layer (aka matrix multiplication), and addition of bias
std::vector<float> fully_connected(const std::vector<float>& input, const std::vector<std::vector<float>>& weights, const std::vector<float>& fully_connected_bias){
    std::vector<float> output(weights[0].size(), 0.0f);

    //Matrix multiplication for weights
    for (int i = 0; i < input.size(); ++i){
        for (int j = 0; j < output.size(); ++j){
            output[j] += input[i]*weights[i][j];
        }
    }

    //Bias addition
    for (int i = 0; i < output.size(); ++i){
        output[i] += fully_connected_bias[i];
    }

    return output;
}