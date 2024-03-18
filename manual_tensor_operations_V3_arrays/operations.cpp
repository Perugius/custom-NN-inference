#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

// DEALLOCATION FUNCTION
template<typename T>
void deallocate_array(T**& array, int first_dim_size) {
    for(int i = 0; i < first_dim_size; ++i) {
        delete[] array[i]; // delete sub array
    }
    delete[] array; // delete outer array
    array = nullptr; // avoid dangling pointer issues
}

// performs the depthwise convolution part of the separable convolution and outputs a vector formated (channel, convolution_result)
float** depthwise_conv1d(float** input, const std::vector<std::vector<float>>& depthwise_weights, int stride, int& input_length, int& num_of_channels){
    num_of_channels = static_cast<int>(depthwise_weights[0].size());
    int kernel_size = static_cast<int>(depthwise_weights.size());
    int single_channel_result_size = static_cast<int>(std::floor(input_length - kernel_size)/stride) +1;
    // allocate result array
    float** result = new float*[num_of_channels];
    for(int i = 0; i < num_of_channels; ++i) {
        result[i] = new float[single_channel_result_size]();
    }

    // biggest loop is channels i.e convolution of first channel then second channel
    for(int channel = 0; channel < num_of_channels; ++channel){
        // second loop iterates over result matrix adding each convolution result
        //std::cout << "CHANNEL" << channel << ": ";
        for(int result_index = 0; result_index < single_channel_result_size; ++result_index){
            // third loop iterates over all the kernels and performs the convolution itself
            for(int kernel_index = 0; kernel_index < kernel_size; ++kernel_index){
                int input_index = result_index * stride + kernel_index;
                if(input_index < input_length) { // Check for out of bounds
                    //std::cout << input[input_index][channel] << '*' << depthwise_weights[kernel_index][channel] << ", ";
                    result[channel][result_index] += input[input_index][channel] * depthwise_weights[kernel_index][channel];
                }   
            }
        }
        //std::cout << "\n";
    }
    // deallocate input since no longer needed
    deallocate_array(input, input_length);
    // update input length each time to know intermediary representation length
    input_length = single_channel_result_size;
    return result;
}

// performs the pointwise convolution part of the separable convolution. Takes input from the depthwise_conv1d and outputs vector formated (convolution_result, filter)
float** pointwise_conv1d(float** input, const std::vector<std::vector<float>>& pointwise_weights, int& input_length, int& num_of_channels){
    num_of_channels = static_cast<int>(pointwise_weights.size());
    int num_of_filters = static_cast<int>(pointwise_weights[0].size());
    int single_filter_result_size = input_length;

    float** result = new float*[single_filter_result_size];
    for(int i = 0; i < single_filter_result_size; ++i){
        result[i] = new float[num_of_filters]();
    }

    for(int filter = 0; filter < num_of_filters; ++filter){
    // iterate over channels to add up all the results since depthwise convolution outputs x number of vector with x = num_of_channels
        for(int channel = 0; channel < num_of_channels; ++channel){
        // iterate over result matrix adding each convolution result
            for(int result_index = 0; result_index < single_filter_result_size; ++result_index){
                result[result_index][filter] += input[channel][result_index] * pointwise_weights[channel][filter];
            }
        }
    }
    num_of_channels = num_of_filters;
    return result;
}

// combine depthwise and pointwise convolution, then deallocate depthwise result before returning complete result
float** separable_conv1d(float** input, const std::vector<std::vector<float>>& depthwise_weights, int stride, int& input_length, int& num_of_channels, const std::vector<std::vector<float>>& pointwise_weights){
    
    // depthwise conv1d
    float** depthwise_result = depthwise_conv1d(input, depthwise_weights, stride, input_length, num_of_channels);
    // save first dim for deallocation
    int depthwise_first_dim_size = num_of_channels;

    //pointwise conv1d
    float** pointwise_result = pointwise_conv1d(depthwise_result, pointwise_weights, input_length, num_of_channels);

    //deallocate and return result
    deallocate_array(depthwise_result, depthwise_first_dim_size);
    return pointwise_result;
}

