from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vit_b_16
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import json

df = pd.read_csv('annotations.csv')
data = []
for _, row in df.iterrows():
    data.append({
        "image_path": row['image_path'],
        "bbox": [row['x_start'], row['y_start'], row['x_end'], row['y_end']],
        "label": row['class_label']
    })

# Dataset
class VisionDataset(Dataset):
    def __init__(self, dataframe, transform=None, target_size=(224, 224)):
        self.dataframe = dataframe
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['image_path']
        bbox = torch.tensor([row['x_start'], row['y_start'], row['x_end'], row['y_end']])
        label = torch.tensor(row['class_label'])

        img = Image.open(img_path).convert("RGB")
        original_size = img.size

        width_ratio = self.target_size[0] / original_size[0]
        height_ratio = self.target_size[1] / original_size[1]
        bbox = bbox * torch.tensor([width_ratio, height_ratio, width_ratio, height_ratio])

        if self.transform:
            img = self.transform(img)

        return img, bbox, label
# Split dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define training and validation datasets
train_dataset = VisionDataset(train_df, transform=transform)
val_dataset = VisionDataset(val_df, transform=transform)

# Define DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Dataset and dataloader
dataset = VisionDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Vision Transformer Model
class VisionTransformerModel(nn.Module):
    def __init__(self, num_classes):
        super(VisionTransformerModel, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()
        self.class_head = nn.Linear(768, num_classes)
        self.bbox_head = nn.Linear(768, 4)
        # Fine tuning: freezing layers
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.vit(x)
        class_out = self.class_head(x)
        bbox_out = self.bbox_head(x)
        return class_out, bbox_out

# Initialize model
num_classes = df['class_label'].nunique()
model = VisionTransformerModel(num_classes=num_classes)

device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model.to(device)

for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

# Loss functions
classification_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.SmoothL1Loss()

# Optimizer
optimizer = optim.Adam([
    {'params': model.class_head.parameters(), 'lr': 0.001},
    {'params': model.bbox_head.parameters(), 'lr': 0.001},
    {'params': filter(lambda p: p.requires_grad, model.vit.parameters()), 'lr': 0.0001}
])

scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
# IoU calculation
def calculate_iou(pred_box, gt_box):
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[2], gt_box[2])
    y2 = min(pred_box[3], gt_box[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = pred_area + gt_area - intersection
    if union == 0:
        return 0

    return intersection / union if union > 0 else 0

# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    total_iou = 0
    num_samples = 0

    with torch.no_grad():
        for images, bboxes, labels in train_loader:
            images, bboxes, labels = images.to(device), bboxes.to(device), labels.to(device)
            class_out, bbox_out = model(images)

            _, predicted = torch.max(class_out, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            for pred_box, gt_box in zip(bbox_out, bboxes):
                iou = calculate_iou(pred_box.cpu().numpy(), gt_box.cpu().numpy())
                total_iou += iou
                num_samples += 1

    classification_accuracy = 100 * correct / total
    mean_iou = total_iou / num_samples if num_samples > 0 else 0

    return classification_accuracy, mean_iou


# # Training loop with early stopping
# early_stopping_patience = 10
# early_stopping_counter = 0
# best_accuracy = 0
# results = {"classification_accuracy": [], "mean_iou": []}

# # Fine tune model
# epochs_stage1 = 100
# for epoch in range(epochs_stage1):
#     model.train()
#     for images, bboxes, labels in dataloader:
#         images, bboxes, labels = images.to(device), bboxes.to(device), labels.to(device)
#         class_out, bbox_out = model(images)
#         class_loss = classification_loss_fn(class_out, labels)
#         bbox_loss = bbox_loss_fn(bbox_out, bboxes)
#         total_loss = class_loss + bbox_loss

#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()
#     # Evaluate after each epoch
#     classification_accuracy, mean_iou = evaluate_model(model, val_loader, device)

#     # Append metrics to results
#     results["classification_accuracy"].append(classification_accuracy)
#     results["mean_iou"].append(mean_iou)

#     # Log results to a JSON file
#     with open("training_metrics.json", "w") as f:
#         json.dump(results, f)

#     print(f"Epoch {epoch + 1}/{epochs_stage1}, "
#         f"Classification Accuracy: {classification_accuracy:.2f}%, Mean IoU: {mean_iou:.4f}")
    
#     if classification_accuracy > best_accuracy:
#         best_accuracy = classification_accuracy
#         torch.save(model.state_dict(), 'best_model_visionT.pth')
#         print(f"Saved model")

#         early_stopping_counter = 0
#     else:
#         early_stopping_counter += 1

#     if early_stopping_counter >= early_stopping_patience:
#         print("Early stopping")
#         break

model.load_state_dict(torch.load('best_model_visionT.pth', map_location=torch.device('cpu')))
print("Saved as best_model_visionT.pth")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

CLASS_NAMES = ["left", "left_shoot", "shoot", "right", "right_shoot"]

def evaluate_test_set(model, test_loader, device):
    """Evaluates the model on the test set and returns true and predicted labels."""
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, bboxes, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            class_out, _ = model(images)
            _, predicted = torch.max(class_out, 1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    return np.array(true_labels), np.array(pred_labels)

true_labels, pred_labels = evaluate_test_set(model, val_loader, device)

valid_indices = (true_labels < 5) & (pred_labels < 5)
true_labels = true_labels[valid_indices]
pred_labels = pred_labels[valid_indices]

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, pred_labels, labels=range(5))

conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_normalized, display_labels=CLASS_NAMES)
disp.plot(cmap=plt.cm.Blues, values_format=".2f")
plt.title("Normalized Confusion Matrix")
plt.show()
