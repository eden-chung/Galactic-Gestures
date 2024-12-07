# Galactic-Gestures

This project involves developing a real-time gesture detection and classification algorithm to control the classic game *Space Invaders* through live video feed. We’ll design a Convolutional Neural Network (CNN) as the model architecture to accurately classify hand gesture directions, enabling intuitive, gesture-based gameplay. 

We will be using [Lee Robinson's Space Invaders](https://github.com/leerob/space-invaders) game, which is a PyGame implementation of Space Invaders. We add our real-time gesture detection model as the main input for the player's movement. 

## Inspiration:

We were inspired by the traditional space invaders game. We thought creating a real-time gesture detection algorithm with our own data would be great for improving motor skills and coordination but it is also a fun game! Another important part of the reason why we decided to choose this game is because it can help with rehabilitation purposes, helping individuals recovering from injuries or with mobility challenges regain motor function.

## How to play:

- **Class 0, Move left**: pointing left
- **Class 1, Move left and shoot**: pointing left and thumb up
- **Class 2, Shoot bullets**: point up
- **Class 3, Move right**: point right
- **Class 4, Move right and shoot**: pointing right and thumb up

The objective is to shoot all of the aliens and live! The player receives three lives and they must kill all the aliens before they die, or else they would lose. 
  
## Installations needed:

We highly recommend playing with the MobileNet model because it has the highest accuracy rate (~98%)! Every model needs different added libraries, so if you need to be wary of the specific libraries that you need. We do not recommend vision transformer at all because of the low accuracy rate (~25%) and it does not need any additional libraries installed other than the general libraries listed. 

### General libraries/algorithms:
- [SORT](https://github.com/abewley/sort)
- Tensorflow
- Pytorch
- NumPy
- Pandas
- Matplotlib

### YOLO:
- Ultralytics

### MobileNet:
- CV2 by OpenCV

## Demo
![](https://github.com/eden-chung/Galactic-Gestures/demo.gif)