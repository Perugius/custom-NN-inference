#include <vector>
#include <iostream>
#include "weights.h"
#include "operations.cpp"



int main() {

    // FIRST LAYER 1D CONV
    std::vector<std::vector<float>> separable_conv1d_result = separable_conv1d(input, depthwise_weights, pointwise_weights, stride);

    // std::cout << "pointwise result\n";
    // for (std::vector<float> filter: separable_conv1d_result){
    //     std::cout << "{";
    //     for (float value : filter){
    //         std::cout << value << ", ";
    //     }
    //     std::cout << "}\n";
    // }

    std::vector<std::vector<float>> normalization_result = batch_normalization(separable_conv1d_result, gamma, beta, moving_mean, moving_variance, epsilon);
    
    //relu(normalization_result);

    for (std::vector<float> vec: normalization_result) {
        std::cout << "{";
        for (float value : vec) {
            std::cout << value << ", ";
        }
        std::cout << "}\n";
    }

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