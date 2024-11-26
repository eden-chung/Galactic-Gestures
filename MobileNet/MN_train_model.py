# train_model.py

import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Large_Weights
import albumentations
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import the GestureDataset from the separate module
from gesture_dataset import GestureDataset

INPUT_SIZE = 224  # Change this depending on what model we are using
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'best_model.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Augmentations for training
augmentations = albumentations.Compose([
    albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    albumentations.Rotate(limit=15, p=0.5),
    albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
    albumentations.GaussianBlur(blur_limit=(3, 5), p=0.3),
    albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4)
])

# Data transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def filter_invalid_annotations(dataset):
    valid_indices = []
    for idx in range(len(dataset.annotations)):
        row = dataset.annotations.iloc[idx]
        img_path = os.path.join(dataset.root_dir, row['image_path'])
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        x_start, y_start, x_end, y_end = map(int, [row['x_start'], row['y_start'], row['x_end'], row['y_end']])
        height, width = image.shape[:2]
        
        x_start = max(0, min(x_start, width))
        x_end = max(0, min(x_end, width))
        y_start = max(0, min(y_start, height))
        y_end = max(0, min(y_end, height))

        if x_start >= x_end or y_start >= y_end:
            print(f"Skipping invalid bounding box in image: {img_path}")
            continue

        valid_indices.append(idx)
    
    dataset.annotations = dataset.annotations.iloc[valid_indices].reset_index(drop=True)
    return dataset

if __name__ == '__main__':
    annotations = pd.read_csv('annotations.csv')

    annotations['class_label'] = annotations['class_label'].astype(int)

    train_annotations, temp_annotations = train_test_split(
        annotations, test_size=0.2, random_state=42, stratify=annotations['class_label']
    )
    val_annotations, test_annotations = train_test_split(
        temp_annotations, test_size=0.5, random_state=42, stratify=temp_annotations['class_label']
    )

    train_dataset = GestureDataset(
        annotations=train_annotations,
        root_dir='.',
        transform=transform,
        augmentations=augmentations  
    )
    val_dataset = GestureDataset(
        annotations=val_annotations,
        root_dir='.',
        transform=transform 
    )
    test_dataset = GestureDataset(
        annotations=test_annotations,
        root_dir='.',
        transform=transform  
    )

    # Filter invalid annotations
    train_dataset = filter_invalid_annotations(train_dataset)
    val_dataset = filter_invalid_annotations(val_dataset)
    test_dataset = filter_invalid_annotations(test_dataset)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")

    # Load the pre-trained MobileNet V3 Large model with default weights
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)

    # classifier to match the number of classes
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)

    # Move model to device
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    images, labels = next(iter(train_loader))
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of labels: {labels}")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total_train_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_train_samples += images.size(0)

        epoch_loss = running_loss / total_train_samples
        print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {epoch_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                total_val_samples += images.size(0)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        val_loss /= total_val_samples
        accuracy = 100 * correct / total_val_samples
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)


from sklearn.metrics import classification_report, accuracy_score
import torch

# Load the best saved model
model = models.mobilenet_v3_large(weights=None)  # Load the same model architecture
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)  # Adjust classifier
model.load_state_dict(torch.load(MODEL_SAVE_PATH))  # Load the saved weights
model = model.to(DEVICE)
model.eval()  # Set to evaluation mode

# Testing loop
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=[f"Class {i}" for i in range(NUM_CLASSES)]))
