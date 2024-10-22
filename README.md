<h1>Custom DNN Edge Deployment for Bachelor Thesis</h1>

This repository contains part of the code for my bachelor thesis. Initially I intended to deploy an atrial fibrillation detection DCNN model
using TensorFlow-Lite, however, due to its many limitations (and mainly due to an overallocation of dynamic memory), I had to write this code to run the DCNN 
using pure C++ (no additional NN libraries). The weights were trained using [HALF: Holistic Auto Machine Learning for FPGAs](https://arxiv.org/abs/2106.14771) and subsequently 
saved in the TensorFlow saved format and extracted as C++ arrays.

The code features multiple version of the same concept of conducting inference. From 


ecg_inference contains script for running the model using the arduino IDE, it is based on manual_tensor_operations_V4

manual_tensor_operations V1-V4 contain the code written for normal c++ execution not for arduino execution. 

for V4 (and ecg_inference) the final dense result does not take into consideration the binary sigmoid function.
