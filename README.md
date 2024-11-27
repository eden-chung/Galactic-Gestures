# Galactic-Gestures

This project involves developing a real-time gesture detection and classification algorithm to control the classic game *Space Invaders* through live video feed. We’ll design a Convolutional Neural Network (CNN) as the model architecture to accurately classify hand gesture directions, enabling intuitive, gesture-based gameplay. 

We will be using [Lee Robinson's Space Invaders](https://github.com/leerob/space-invaders) game, which is a PyGame implementation of Space Invaders. We add our real-time gesture detection model as the main input for the player's movement. 

## How to play:

- **Class 0, Move left**: pointing left
- **Class 1, Move left and shoot**: pointing left and thumb up
- **Class 2, Shoot bullets**: point up
- **Class 3, Move right**: point right
- **Class 4, Move right and shoot**: pointing right and thumb up

   
Mobile Net Accuracy Rate RN

Epoch 1/10, Training Loss: 0.9764
Validation Loss: 3.0783, Accuracy: 60.00%

Epoch 2/10, Training Loss: 0.1713
Validation Loss: 0.9356, Accuracy: 85.45%

Epoch 3/10, Training Loss: 0.0872
Validation Loss: 1.5026, Accuracy: 74.55%

Epoch 4/10, Training Loss: 0.1213
Validation Loss: 1.6549, Accuracy: 74.55%

Epoch 5/10, Training Loss: 0.2231
Validation Loss: 0.7125, Accuracy: 81.82%

Epoch 6/10, Training Loss: 0.1095
Validation Loss: 0.3252, Accuracy: 92.73%

Epoch 7/10, Training Loss: 0.0297
Validation Loss: 0.3253, Accuracy: 96.36%

Epoch 8/10, Training Loss: 0.1317
Validation Loss: 0.5529, Accuracy: 94.55%

Epoch 9/10, Training Loss: 0.1593
Validation Loss: 1.4616, Accuracy: 54.55%

Epoch 10/10, Training Loss: 0.0711
Validation Loss: 1.5608, Accuracy: 74.55%
