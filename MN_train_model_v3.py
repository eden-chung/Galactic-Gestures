# train_model.py

import os
import cv2
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Large_Weights
import albumentations
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # Added for visualization

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Constants
INPUT_SIZE = 224  # Input size for MobileNet V3 Large
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 50  # Increased epochs to allow early stopping to take effect
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'best_model_v3.pth'
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pre-trained models
])

# GestureDataset class that uses bounding boxes
class GestureDataset(Dataset):
    def __init__(self, annotations, root_dir, transform=None, augmentations=None):
        self.annotations = annotations.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.augmentations = augmentations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Read the row for the current image
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image_path'])
        
        # Read the image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        
        # Get the bounding box coordinates
        x_start, y_start, x_end, y_end = map(int, [row['x_start'], row['y_start'], row['x_end'], row['y_end']])

        height, width = image.shape[:2]
        x_start = max(0, min(x_start, width))
        x_end = max(0, min(x_end, width))
        y_start = max(0, min(y_start, height))
        y_end = max(0, min(y_end, height))

        # Check if the bounding box is valid
        if x_start >= x_end or y_start >= y_end:
            raise ValueError(f"Invalid bounding box for image {img_path}: x_start={x_start}, x_end={x_end}, y_start={y_start}, y_end={y_end}")

        # Crop the hand region
        hand_crop = image[y_start:y_end, x_start:x_end]
        
        # Check if the hand_crop is empty
        if hand_crop.size == 0:
            raise ValueError(f"Empty hand crop for image {img_path} with bounding box {x_start}, {y_start}, {x_end}, {y_end}")

        # Convert BGR to RGB
        hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
        
        if self.augmentations:
            augmented = self.augmentations(image=hand_crop)
            hand_crop = augmented['image']
        
        if self.transform:
            hand_crop = self.transform(hand_crop)
        else:
            hand_crop = transforms.ToTensor()(hand_crop)
        
        label = int(row['class_label'])
        return hand_crop, label

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

# Visualization function
def visualize_dataset_sample(dataset, idx):
    hand_crop, label = dataset[idx]
    
    row = dataset.annotations.iloc[idx]
    img_path = os.path.join(dataset.root_dir, row['image_path'])
    image = cv2.imread(img_path)
    
    x_start, y_start, x_end, y_end = map(int, [row['x_start'], row['y_start'], row['x_end'], row['y_end']])
    
    image_with_bbox = image.copy()
    cv2.rectangle(image_with_bbox, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    image_with_bbox = cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB)
    
    hand_crop_image = hand_crop.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
    
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    hand_crop_image = std * hand_crop_image + mean
    hand_crop_image = np.clip(hand_crop_image, 0, 1)
    hand_crop_image = (hand_crop_image * 255).astype('uint8')
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image_with_bbox)
    axs[0].set_title('Original Image with Bounding Box')
    axs[0].axis('off')
    
    axs[1].imshow(hand_crop_image)
    axs[1].set_title(f'Cropped Hand Region - Label: {label}')
    axs[1].axis('off')
    
    plt.show()

if __name__ == '__main__':
    set_seed()

    # Load annotations
    annotations = pd.read_csv('annotations.csv')
    annotations['class_label'] = annotations['class_label'].astype(int)

    # Split into train, validation, and test sets
    train_annotations, temp_annotations = train_test_split(
        annotations, test_size=0.2, random_state=42, stratify=annotations['class_label']
    )
    val_annotations, test_annotations = train_test_split(
        temp_annotations, test_size=0.5, random_state=42, stratify=temp_annotations['class_label']
    )

    test_annotations.to_csv('test_annotations.csv', index=False)

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

    train_dataset = filter_invalid_annotations(train_dataset)
    val_dataset = filter_invalid_annotations(val_dataset)

    print("Visualizing training samples:")
    for idx in range(3):  
        visualize_dataset_sample(train_dataset, idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)

    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    best_accuracy = 0.0
    epochs_no_improve = 0
    n_epochs_stop = 3  # Number of epochs to wait before early stopping

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

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with validation loss: {val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s)")

        # Early stopping
        if epochs_no_improve >= n_epochs_stop:
            print(f"Early stopping after {epoch + 1} epochs")
            break

        # Optionally, save the model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f"best_model_acc_{accuracy:.2f}.pth")
            print(f"New best model saved with accuracy: {accuracy:.2f}%")