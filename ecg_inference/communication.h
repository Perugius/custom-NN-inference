#ifndef COMMUNICATION_H
#define COMMUNICATION_H

//Send array over uart
void send_array(float* input, unsigned int input_length);

float* receive_array_V2(int input_length);
#endif // COMMUNICATION_H