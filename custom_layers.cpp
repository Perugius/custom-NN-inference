#include <iostream>
#include <vector>
#include <algorithm>

std::vector<float> single_filter_1Dconvolution(const std::vector<float>& input, const std::vector<std::vector<float>>& kernel, int input_length, int num_of_inputs, int stride) {

    int input_size = static_cast<int>(input_length);
    int kernel_size = static_cast<int>(kernel[0].size());
    int result_size = input_size- kernel_size + 1;
    std::vector<float> result(result_size, 0.0);
    // take one of the input vectors and perform convolution and add it to the result
    std::vector<float> single_input(input_length, 0.0);

    for (int k = 0; k < num_of_inputs; ++k){
        // iterate over vectors because input is flat so a (80, 3) shape will be represented as (240) -> iterate over 3 vectors of length 80
        // each of those single vectors is saved into single_input
        std::copy(input.begin() + k*input_length, input.begin() + k*input_length + input_length, single_input.begin());

        // i iterates over result (i.e kernel sliding over input vector), j iterates over the kernel itself multiplying and adding into the results
        // finally k iterates over all the kernels and adds up everything to the same result
        for (int i = 0; i < result_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                result[i] += single_input[i + j] * kernel[k][j];
            }
        }
    }
    return result;
}

//uses single_filter_IDconvolution function to convolute over multiple filters
std::vector<std::vector<float>> multi_filter_1Dconvolution(const std::vector<float>& input, std::vector<std::vector<std::vector<float>>>& filters, int input_length, int num_of_inputs, int stride, int num_of_filters){
    std::vector<std::vector<float>> result_multi;
    //iterate over all filters passing the kernels to the 1dconv function

    for (int i = 0; i < num_of_filters; ++i){
        std::vector<float> single_result = single_filter_1Dconvolution(input, filters[i], input_length, num_of_inputs, 1);
        result_multi.push_back(single_result);
    }

    return result_multi;
}


//VARIABLES THAT HAVE TO BE CHANGED MANUALLY: stride (for now only 1 works), input (its the whole flat!! input to the conv1d, the reshaping to appropriate size, same as trained, happens in the 1dconv function), input_length (depends on how the input is dimensioned)
//number of filters and the kernels that go into them (i.e filters.push_back(kernel1)...), and all the kernels (write python script to auto generate the kernels)
int main() {

    //----------------------------------------------------------INPUTS------------------------------------------------------------
    // input test vector of size 240
    std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int stride = 1;
    int num_of_filters = 2;
     // dimension of conv1d input, i.e if trained on (80, 3) dims then input_length = 80 and num_of_inputs = 3
    int input_length = 10;
    int num_of_inputs = static_cast<int>(input.size()/input_length);
    // each vector in the kernel represents the 2nd dimention of the input, i.e 3x3 kernel means the input shape is (x, 3), a 3x4 kernel means (x, 4 input)
    std::vector<std::vector<float>> kernel1 = {
        {0, 1, 0},
        {0, 2, 0},
        {0, 3, 0},
    };
    std::vector<std::vector<float>> kernel2 = {
        {5, 1, 3},
        {0, 2, 0},
        {0, 0, 0},
    };

    std::vector<std::vector<std::vector<float>>> filters;
    filters.push_back(kernel1);
    filters.push_back(kernel2);

    //----------------------------------------------------END INPUTS-------------------------------------------------

    // check if amount of kernels matches input dimensions
    if (kernel1.size() != num_of_inputs){
        std::cout << "ERROR: Filter dimension does not match input dimension";
        return 1;
    }

    //std::vector<float> result = single_filter_1Dconvolution(input, kernel1, input_length, num_of_inputs, 1);
    std::vector<std::vector<float>> result = multi_filter_1Dconvolution(input, filters, input_length, num_of_inputs, stride, num_of_filters);



    std::cout << "convolution result ";
    for (std::vector<float> single_filter_result : result){
        std::cout << "filter";
        for (float value : single_filter_result) {
            std::cout << value << " ";
    }
    }


    // Print the result
    std::cout << std::endl;

    return 0;
}