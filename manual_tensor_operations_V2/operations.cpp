#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

// performs the depthwise convolution part of the separable convolution and outputs a vector formated (channel, convolution_result)
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

// performs the pointwise convolution part of the separable convolution. Takes input from the depthwise_conv1d and outputs vector formated (convolution_result, filter)
std::vector<std::vector<float>> pointwise_conv1d(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& pointwise_weights){
    int num_of_channels = static_cast<int>(pointwise_weights.size());
    int num_of_filters = static_cast<int>(pointwise_weights[0].size());
    int single_filter_result_size(input[0].size());

    std::vector<std::vector<float>> result(single_filter_result_size, std::vector<float>(num_of_filters, 0.0));

    for(int filter = 0; filter < num_of_filters; ++filter){
    // iterate over channels to add up all the results since depthwise convolution outputs x number of vector with x = num_of_channels
        for(int channel = 0; channel < num_of_channels; ++channel){
        // iterate over result matrix adding each convolution result
            for(int result_index = 0; result_index < single_filter_result_size; ++result_index){
                result[result_index][filter] += input[channel][result_index] * pointwise_weights[channel][filter];
            }
        }
    }
    return result;
}

// combine depthwise and pointwise to do complete separable 1d convolution operation
std::vector<std::vector<float>> separable_conv1d(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& depthwise_weights, const std::vector<std::vector<float>>& pointwise_weights, int stride){
    std::vector<std::vector<float>> depthwise_conv1d_result = depthwise_conv1d(input, depthwise_weights, stride);
    std::vector<std::vector<float>> pointwise_conv1d_result = pointwise_conv1d(depthwise_conv1d_result, pointwise_weights);
    return pointwise_conv1d_result;
}

// flatten function
std::vector<float> flatten(std::vector<std::vector<float>>& input){
    std::vector<float> output;

    for (const auto& values : input){
        output.insert(output.end(), values.begin(), values.end());
    }
    return output;
}

//dense layer (aka matrix multiplication), and addition of bias
std::vector<float> fully_connected(const std::vector<float>& input, const std::vector<std::vector<float>>& weights){//, const std::vector<float>& bias){
    std::vector<float> output(weights[0].size(), 0.0f);

    //Matrix multiplication for weights
    for (int i = 0; i < input.size(); ++i){
        for (int j = 0; j < output.size(); ++j){
            output[j] += input[i]*weights[i][j];
        }
    }

    // //Bias addition
    // for (int i = 0; i < output.size(); ++i){
    //     output[i] += fully_connected_bias[i];
    // }

    return output;
}

std::vector<std::vector<float>> batch_normalization(const std::vector<std::vector<float>>& input,
                        const std::vector<float>& gamma,
                        const std::vector<float>& beta,
                        const std::vector<float>& moving_mean,
                        const std::vector<float>& moving_variance,
                        float epsilon) {
                            
    int single_filter_result_size = static_cast<int>(input.size());
    int num_of_filters = static_cast<int>(input[0].size()); // Assuming gamma size is equal to the number of filters

    // Batch normalization output
    std::vector<std::vector<float>> output(single_filter_result_size, std::vector<float>(num_of_filters, 0.0));

    for (int result_index = 0; result_index < single_filter_result_size; ++result_index) {
        for (int filter = 0; filter < num_of_filters; ++filter) {
            // Normalize
            float norm = (input[result_index][filter] - moving_mean[filter]) / std::sqrt(moving_variance[filter] + epsilon);
            // Scale and shift
            output[result_index][filter] = gamma[filter] * norm + beta[filter];
        }
    }
    return output;
}

// MODIFIES THE INPUT VECTOR ITSELF, DOES NOT RETURN A VECTOR!!!
void relu(std::vector<std::vector<float>>& input){
    //std::vector<std::vector<float>> output = input;
    for (int i = 0; i < input.size(); ++i){
        for (int j = 0; j < input[i].size(); ++j){
            if(input[i][j] < 0){input[i][j] = 0;}
        }
    }
}