<h1>Custom DNN Edge Deployment for Bachelor Thesis</h1>

This repository contains part of the code for my bachelor thesis. Initially I intended to deploy an atrial fibrillation detection DCNN model
using TensorFlow-Lite, however, due to its many limitations (and mainly due to an overallocation of dynamic memory), I had to write this code to run the DCNN 
using pure C++ (no additional NN libraries). The weights were trained using [HALF: Holistic Auto Machine Learning for FPGAs](https://arxiv.org/abs/2106.14771) and subsequently 
saved in the TensorFlow saved format and extracted as C++ arrays.

<h2>Versions</h2>

The code features multiple version of the same concept of conducting inference:
- V1 : Uses std::vector<std::vector<float>>, aka nested vectors (2D) for both the weights and intermediary results and performs standards 1D Convolution
- V2 : Also uses std::vector<std::vector<float>>, but instead performs Separable 1D Convolution
- V3 : Uses 2D lists ([][]), performs Separable 1D Convolution
- V4 : Uses 1D lists ([]), performs Separable 1D Convolution, final dense layer does not include binary sigmoid function and has to be added manually
- ecg_inference : same as V4 just targeted towards deployment via Arduino IDE, same with above with dense layer. Additionally uses custom UART communication code to automatically test large numbers of input arrays, since only 1 inference at a time can be performed.

Each iteration changed to further optimize the inference, with only V4 being able to run our somewhat large models on very limited edge devices (i.e Raspberry Pi Pico W), since 1D lists are by far the 
most efficient method to store the intermediary results. The weights are in comparison very few in number so they are stored in std::vector<> form with no real performance penalty.

The layers have to be manually ordered according to the shape of the saved model. The weights have to be added manually in the weights.h file.
