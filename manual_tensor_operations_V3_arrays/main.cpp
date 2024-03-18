#include <vector>
#include <iostream>
#include "weights.h"
#include "operations.cpp"
#include "input.h"


int main() {

    int input_length = first_input_length;
    int num_of_channels = first_num_of_channels;
    // FIRST LAYER 1D CONV input
    float** input = new float*[input_length];
    for (int i = 0; i < input_length; ++i){
        input[i] = new float[num_of_channels]();
    }

    //copy contenst of the first input to input array
    for(int i = 0; i < input_length; ++i){
        for(int j = 0; j < num_of_channels; ++j){
            input[i][j] = test_data[i][j];
        }
    }
    

    // float** depthwise_result = depthwise_conv1d(input, depthwise_weights, stride, input_length, num_of_channels);
    // // for deallocation!
    // int depthwise_first_dim_size = num_of_channels;

    // float** pointwise_result = pointwise_conv1d(depthwise_result, pointwise_weights, input_length, num_of_channels);

    float** separable_conv1d_result = separable_conv1d(input, depthwise_weights, stride, input_length, num_of_channels, pointwise_weights);
    // for deallocation
    int separable_first_dim_size = input_length;

    //deallocate_array(depthwise_result, depthwise_first_dim_size);

    std::cout << "conv1d result, output length:\n" << input_length;
    for (int i = 0; i < input_length; ++i){
        std::cout << "{";
        for (int j = 0; j < num_of_channels; ++j){
            std::cout << separable_conv1d_result[i][j] << ", ";
        }
        std::cout << "}\n";
    }


    //std::vector<std::vector<float>> normalization_result = batch_normalization(separable_conv1d_result, gamma, beta, moving_mean, moving_variance, epsilon);
    
    //relu(normalization_result);

    // for (std::vector<float> vec: normalization_result) {
    //     std::cout << "{";
    //     for (float value : vec) {
    //         std::cout << value << ", ";
    //     }
    //     std::cout << "}\n";
    // }

    // SECOND LAYER FLATTEN

    // std::vector<float> flatten_result = flatten(separable_conv1d_result);
    // std::cout << "flatten result = {\n";
    // for (float value : flatten_result){
    //     std::cout << value << ", ";
    // }
    // std::cout << "}\n";

    // // THIRD LAYER DENSE (FINAL LAYER)
    // std::vector<float> final_result = fully_connected(flatten_result, dense_weights);

    // std::cout << "final result = {\n";
    // for (float value : final_result){
    //     std::cout << value << ", ";
    // }
    // std::cout << "}\n";

    // return 0;
}