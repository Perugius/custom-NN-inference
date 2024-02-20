#include <iostream>
#include <vector>
#include <algorithm>

std::vector<float> convolution(const std::vector<float>& input, const std::vector<std::vector<float>>& kernel, int input_length, int num_of_inputs, int stride) {

    int input_size = static_cast<int>(input_length);
    int kernel_size = static_cast<int>(kernel[0].size());
    int result_size = input_size- kernel_size + 1;
    std::vector<float> result(result_size, 0.0);

    // take one of the input vectors and perform convolution and add it to the result
    std::vector<float> single_input(input_length, 0.0);


    for (int k = 0; k < num_of_inputs; ++k){
        std::copy(input.begin() + k*input_length, input.begin() + k*input_length + input_length, single_input.begin());

        for (int i = 0; i < result_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                result[i] += single_input[i + j] * kernel[k][j];
            }
        }
    }
    return result;
}

//VARIABLES TO MANUALLY CHANGE: stride (for now only 1 works), input (its the whole flat!! input to the conv1d, the reshaping to appropriate size, same as trained, happens in the function), input_length (depends on how the input is dimensioned)

int main() {
    // input test vector of size 240
    std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    int stride = 1;
     // dimension of conv1d input, i.e if trained on (80, 3) dims then input_length = 80 and num_of_inputs = 3
    int input_length = 10;
    int num_of_inputs = static_cast<int>(input.size()/input_length);

    // each vector in the kernel represents the 2nd dimention of the input, i.e 3x3 kernel means the input shape is (x, 3), a 3x4 kernel means (x, 4 input)
    std::vector<std::vector<float>> kernel = {
        {0, 1, 0},
        {0, 2, 0},
        {0, 3, 0}
    };

    std::vector<float> result = convolution(input, kernel, input_length, num_of_inputs, 1);

    std::cout << "convolution result ";
    for (float value : result) {
        std::cout << value << " ";
    }
    

    // Print the result
    std::cout << std::endl;

    return 0;
}