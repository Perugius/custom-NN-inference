#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

std::vector<std::vector<float>> depthwise_conv1d(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& depthwise_weights, int stride){
    int num_of_channels = static_cast<int>(depthwise_weights[0].size());
    int channel_length = static_cast<int>(input.size());
    int kernel_size = static_cast<int>(depthwise_weights.size());
    int single_channel_result_size = static_cast<int>(std::floor(channel_length - kernel_size)/stride) +1;

    std::vector<std::vector<float>> result(num_of_channels, std::vector<float>(single_channel_result_size, 0.0));

    // biggest loop is channels i.e convolution of first channel then second channel
    for(int channel = 0; channel < num_of_channels; ++channel){
        // second loop iterates over result matrix adding each convolution result
        for(int result_index = 0; result_index < single_channel_result_size; ++result_index){
            // third loop iterates over all the kernels and performs the convolution itself
            for(int kernel_index = 0; kernel_index < kernel_size; ++kernel_index){
                result[channel][result_index] += input[result_index * stride + kernel_index][channel] * depthwise_weights[kernel_index][channel];
            }   
        }
    }
    return result;
}