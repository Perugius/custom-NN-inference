#include "Arduino.h"


void send_array(float* input, unsigned int input_length){
  
  unsigned int input_bytes = input_length*sizeof(float);

  // Signal transmission start and send number of bytes to be transmitted
  Serial.print("SENDING ");
  Serial.println(input_bytes);
  delay(100);
  // Transmit array
  Serial.write((const byte*)input, input_bytes);
  bool ack_received = false;
  while (!ack_received){
    // Wait for ack signal
    String received_message = Serial.readStringUntil('\n');
    if (received_message == "ACK"){
      ack_received = true;
    }
  }
  //delete[] input;
}

float* receive_array_V2(int input_length){
  unsigned int input_bytes = input_length*sizeof(float);

  float* array = new float[input_length];
  if (!array) {
    Serial.println("Memory allocation failed");
    return NULL;
  }
  // Signal transmission start and send number of bytes to be transmitted
  Serial.print("REQUESTING ");
  Serial.println(1);
  delay(50);

  for (int i = 0; i < input_length; ++i){
    delay(30);
    if (Serial.available() >= sizeof(float)){
      union {
        byte bytes[4];
        float value;
      } floatUnion;

      for (int j = 0; j < 4; j++){
        floatUnion.bytes[j] = Serial.read();
      }

      array[i] = floatUnion.value;
    }
  }
  return array;
}



