#include <vector>
#include <iostream>
#include "weights.h"
#include "operations.cpp"
#include "input.h"

int main() {
    
    int input_length = first_input_length;
    int num_of_channels = first_num_of_channels;

    // load input vector into flat array
    float* input = new float[input_length*num_of_channels]();
    for (int i = 0; i < input_length; ++i) {
        for (int j = 0; j < num_of_channels; ++j) {
            input[i * num_of_channels + j] = test_data[i][j];
        }
    }

    //perform separable convolution
    float* depthwise_result = depthwise_conv1d(input, depthwise_weights, stride, input_length, num_of_channels);
    float* pointwise_result = pointwise_conv1d(depthwise_result, pointwise_weights, input_length, num_of_channels);
    batch_normalization(pointwise_result, gamma_coeff, beta, moving_mean, moving_variance, input_length, num_of_channels, epsilon);
    relu(pointwise_result, input_length, num_of_channels);
    
    for(int i = 0; i < 66; ++i){
        std::cout << pointwise_result[i] << ", ";
    }



    // for(int i = 0; i < 10; ++i){
    //     std::cout << pointwise_result[i] << ", ";
    // }

    // delete[] pointwise_result;

}
