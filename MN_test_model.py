# test_model.py

import os
import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Large_Weights
from sklearn.metrics import classification_report, accuracy_score

# Import the GestureDataset from your separate module
from gesture_dataset import GestureDataset

# Constants
INPUT_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5
MODEL_SAVE_PATH = 'best_model_MN.pth'  # Ensure this matches the saved model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def filter_invalid_annotations(dataset):
    # [Function code remains the same]
    valid_indices = []
    for idx in range(len(dataset.annotations)):
        row = dataset.annotations.iloc[idx]
        img_path = os.path.join(dataset.root_dir, row['image_path'])
        image = cv2.imread(img_path)
        if image is None:
            continue
        x_start, y_start, x_end, y_end = map(int, [row['x_start'], row['y_start'], row['x_end'], row['y_end']])
        height, width = image.shape[:2]
        x_start = max(0, min(x_start, width))
        x_end = max(0, min(x_end, width))
        y_start = max(0, min(y_start, height))
        y_end = max(0, min(y_end, height))
        if x_start >= x_end or y_start >= y_end:
            continue
        valid_indices.append(idx)
    dataset.annotations = dataset.annotations.iloc[valid_indices].reset_index(drop=True)
    return dataset

if __name__ == '__main__':
    # Load test annotations
    test_annotations = pd.read_csv('test_annotations.csv')
    test_annotations['class_label'] = test_annotations['class_label'].astype(int)

    # Create test dataset
    test_dataset = GestureDataset(
        annotations=test_annotations,
        root_dir='.',
        transform=transform  # No augmentations for testing
    )

    # Filter invalid annotations
    test_dataset = filter_invalid_annotations(test_dataset)

    # Create DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Load the best saved model
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

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
    class_names = [f"Class {i}" for i in range(NUM_CLASSES)]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))