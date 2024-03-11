#include <vector>
#include <iostream>
#include "weights.h"
#include "operations.cpp"



int main(){

    std::vector<std::vector<float>> conv1d_result = depthwise_conv1d(input, depthwise_weights, stride);

    std::cout << "depthwise result\n";
    for (std::vector<float> channel : conv1d_result){
        std::cout << "channel {";
        for (float value : channel) {
            std::cout << value << " ";
        }
        std::cout << "}";
    }
}