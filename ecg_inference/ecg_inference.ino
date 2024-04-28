#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "weights.h"
#include "operations.h"
#include "input.h"
#include "Arduino.h"
#include "communication.h"

unsigned endTime;
unsigned startTime;
unsigned timeElapsed;

float final_result_arr[2] = {};
void setup() {
  Serial.begin(115200);
  while(!Serial){
    ;
  }
  delay(100);
  //Serial.println("SERIAL CONNECTED");
}

void loop() {

    
    int input_length = first_input_length;
    int num_of_channels = first_num_of_channels;

    float* input = receive_array_V2(input_length * num_of_channels);
    // load input vector into flat array
    // float* input = new float[input_length*num_of_channels]();
    // for (int i = 0; i < input_length * num_of_channels; ++i){
    //     input[i] = test_data[i];
    // }

    startTime = micros();

    // perform separable convolution 1ST LAYER
    float* depthwise_result_01 = depthwise_conv1d(input, depthwise_weights_01, stride_01, input_length, num_of_channels);
    float* pointwise_result_01 = pointwise_conv1d(depthwise_result_01, pointwise_weights_01, input_length, num_of_channels);
    batch_normalization(pointwise_result_01, gamma_coeff_01, beta_01, moving_mean_01, moving_variance_01, input_length, num_of_channels, epsilon);
    relu(pointwise_result_01, input_length, num_of_channels);

    // perform separable convolution 2ND LAYER
    float* depthwise_result_02 = depthwise_conv1d(pointwise_result_01, depthwise_weights_02, stride_02, input_length, num_of_channels);
    float* pointwise_result_02 = pointwise_conv1d(depthwise_result_02, pointwise_weights_02, input_length, num_of_channels);
    batch_normalization(pointwise_result_02, gamma_coeff_02, beta_02, moving_mean_02, moving_variance_02, input_length, num_of_channels, epsilon);
    relu(pointwise_result_02, input_length, num_of_channels);

    // perform global pooling before dense layer
    float* global_pooling_result = global_pooling1D(pointwise_result_02, input_length, num_of_channels);

    // performs final dense layer
    float* fully_connected_result = fully_connected(global_pooling_result, dense_weights, input_length, num_of_channels);
    
    float fully_connected_classification = fully_connected_result[0];
    // apply sigmoid activation
    float final_result = 1.0 / (1.0 + exp(-fully_connected_classification));

    final_result_arr[0] = final_result; 
    send_array(final_result_arr, 2);
    // endTime = micros();
    // timeElapsed = endTime - startTime; 

    //Serial.println(1000*final_result);
    //Serial.println(timeElapsed);
    delay(1000);
    delete[] fully_connected_result;

}