# Galactic-Gestures

This project involves developing a real-time gesture detection and classification algorithm to control the classic game *Space Invaders* through live video feed. We’ll design a Convolutional Neural Network (CNN) as the model architecture to accurately classify hand gesture directions, enabling intuitive, gesture-based gameplay. 

We will be using [Lee Robinson's Space Invaders](https://github.com/leerob/space-invaders) game, which is a PyGame implementation of Space Invaders. We add our real-time gesture detection model as the main input for the player's movement. 

## How to play:

- **Class 0, Move left**: pointing left
- **Class 1, Move left and shoot**: pointing left and thumb up
- **Class 2, Shoot bullets**: point up
- **Class 3, Move right**: point right
- **Class 4, Move right and shoot**: pointing right and thumb up

   