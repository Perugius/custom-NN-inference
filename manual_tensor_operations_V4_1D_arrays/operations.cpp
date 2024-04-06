#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>



// performs the depthwise convolution part of the separable convolution
// outputs flat array where each block of single_channel_result_size is the equivalent channels result
// i.e result = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5}, if single_channel_result_size = 5 then the first 5 elements are channel 1 results, the next 5 are channel 2, and so on
float* depthwise_conv1d(float* input, const std::vector<std::vector<float>>& depthwise_weights, int stride, int& input_length, int& num_of_channels){
    num_of_channels = static_cast<int>(depthwise_weights[0].size());
    int kernel_size = static_cast<int>(depthwise_weights.size());
    int single_channel_result_size = static_cast<int>(std::floor(input_length - kernel_size)/stride) +1;

    // allocate result array, initialize to 0
    float* result = new float[num_of_channels*single_channel_result_size]();

    // first loop iterates over channels i.e if we have 2 channels then we access element 0, 2, 4 etc. after first loop we access 1, 3, 5 etc
    for(int channel = 0; channel < num_of_channels; ++channel){
        for(int result_index = 0; result_index < single_channel_result_size; ++result_index){
            for(int kernel_index = 0; kernel_index < kernel_size; ++kernel_index){
                int input_index = (result_index * stride + kernel_index) * num_of_channels + channel;
                if (input_index < input_length * num_of_channels){
                    result[result_index + (channel * single_channel_result_size)] += input[input_index] * depthwise_weights[kernel_index][channel];
                }
            }
        }   
    }
    // deallocate input since no longer needed
    delete[] input;
    // update input length each time to know intermediary representation length
    // num_of_channels stays the same
    input_length = single_channel_result_size;
    return result;
}

// performs the pointwise convolution part of the separable convolution
// outputs flat array where shape corresponds to flattened array of shape ()
float* pointwise_conv1d(float*input, const std::vector<std::vector<float>>& pointwise_weights, int& input_length, int& num_of_channels){
    if(num_of_channels != pointwise_weights.size()){std::cout << "NUM OF CHANNELS MISMATCH: pointwise_conv1d";}
    num_of_channels = static_cast<int>(pointwise_weights.size());
    int num_of_filters = static_cast<int>(pointwise_weights[0].size());
    int single_filter_result_size = input_length;

    // allocate result array, initialize to 0
    float* result = new float[num_of_filters*single_filter_result_size]();

    for(int filter = 0; filter < num_of_filters; ++filter){
        for(int channel = 0; channel < num_of_channels; ++channel){
            for(int result_index = 0; result_index < single_filter_result_size; ++result_index){
                result[result_index * num_of_filters + filter] += input[result_index + (channel * single_filter_result_size )] * pointwise_weights[channel][filter];
            }
        }
    }

    // deallocate input since no longer needed
    delete[] input;
    // new number of channels is the same as the amount of filters
    num_of_channels = num_of_filters;
    return result;
}

void batch_normalization(float* input, const std::vector<float>& gamma_coeff, const std::vector<float>& beta, const std::vector<float>& moving_mean, const std::vector<float>& moving_variance, int input_length, int num_of_channels, float epsilon) {

    // Normalize the output
    for (int result_index = 0; result_index < input_length; ++result_index) {
        for (int channel = 0; channel < num_of_channels; ++channel) {
            int index = result_index * num_of_channels + channel; // Corrected indexing
            input[index] = (input[index] - moving_mean[channel]) / std::sqrt(moving_variance[channel] + epsilon) * gamma_coeff[channel] + beta[channel];
        }
    }


}

float* fully_connected(float* input, const std::vector<std::vector<float>>& weights, int& input_length, int& num_of_channels){
    int result_size = weights[0].size();

    // allocate result array, initialize to 0
    float* result = new float[result_size]();

    for (int i = 0; i < input_length * num_of_channels; ++i){
        for (int j = 0; j < result_size; ++j){
            result[j] += input[i]*weights[i][j];
        }
    }

    // deallocate input since no longer needed
    delete[] input;
    // input_length and num_of_channels change for intermediary representation
    input_length = result_size;
    num_of_channels = 1;
    return result;
}

float* global_pooling1D(float* input, int& input_length, int& num_of_channels){
    
    int result_size = num_of_channels;

    // allocate result array, initialize to 0
    float* result = new float[result_size]();
    
    // sum up total in each channel
    for (int channel = 0; channel < num_of_channels; ++channel){
        for(int timestep = 0; timestep < input_length; ++timestep){
            result[channel] += input[timestep * num_of_channels + channel];
        }
    }

    // calculate mean by diving by number of elements in each channel
    for(int channel = 0; channel < num_of_channels; ++channel){
        result[channel] = result[channel] / input_length;
    }

    // deallocate input since no longer needed
    delete[] input;

    // change input_length, num_of_channels remains unchanged
    input_length = 1;
    return result;
}

void relu(float* input, int input_length, int num_of_channels){
    for (int i = 0; i < input_length*num_of_channels; ++i){
        if(input[i] < 0){
            input[i] = 0;
        }
    }
}