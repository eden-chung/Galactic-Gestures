# gesture_dataset.py

import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms

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

        #print(f"Processing image: {img_path}")
        #print(f"Image shape: {image.shape}")
        #print(f"Bounding box coordinates: x_start={x_start}, y_start={y_start}, x_end={x_end}, y_end={y_end}")

        # Ensure coordinates are within image boundaries
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
        
        # Apply augmentations
        if self.augmentations:
            augmented = self.augmentations(image=hand_crop)
            hand_crop = augmented['image']
        
        # Apply transformations
        if self.transform:
            hand_crop = self.transform(hand_crop)
        else:
            hand_crop = transforms.ToTensor()(hand_crop)
        
        label = int(row['class_label'])
        return hand_crop, label