import cv2
import torch
import numpy as np
from torchvision import models, transforms
import torch.nn as nn

INPUT_SIZE = 224
NUM_CLASSES = 5
MODEL_SAVE_PATH = 'V2_30epoch.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [f"Class {i}" for i in range(NUM_CLASSES)]

# Data transformations for live input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path, num_classes):
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model.to(DEVICE).eval()

# Predict gesture from a single frame
def predict_gesture(model, frame):
    with torch.no_grad():
        input_tensor = transform(frame).unsqueeze(0).to(DEVICE)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted_class = torch.max(output, 1)
        return CLASS_NAMES[predicted_class.item()], probabilities[0][predicted_class.item()].item()

# Initialize the model
model = load_model(MODEL_SAVE_PATH, NUM_CLASSES)

# OpenCV for live capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the camera.")
        break

    # Convert BGR to RGB for compatibility with PyTorch
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        # Predict gesture
        predicted_label, confidence = predict_gesture(model, rgb_frame)
        text = f"{predicted_label} ({confidence * 100:.2f}%)"
    except Exception as e:
        text = "Error: Unable to predict."

    # Overlay prediction on the frame
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()