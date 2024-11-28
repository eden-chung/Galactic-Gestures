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

df = pd.read_csv('annotations.csv')
data = []
for _, row in df.iterrows():
    data.append({
        "image_path": row['image_path'],
        "bbox": [row['x_start'], row['y_start'], row['x_end'], row['y_end']],
        "label": row['class_label']
    })

# Output example images (feel free to comment this out)
def visualize_image(row):
    img = Image.open(row['image_path'])
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    x_start, y_start, x_end, y_end = row['x_start'], row['y_start'], row['x_end'], row['y_end']
    width = x_end - x_start
    height = y_end - y_start
    rect = patches.Rectangle((x_start, y_start), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.title(f"Class: {row['class_label']}")
    plt.axis('off')
    plt.show()

for i in range(3):
    visualize_image(df.iloc[i])

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

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

    def forward(self, x):
        x = self.vit(x)
        class_out = self.class_head(x)
        bbox_out = self.bbox_head(x)
        return class_out, bbox_out

# Initialize model
num_classes = df['class_label'].nunique()
model = VisionTransformerModel(num_classes=num_classes)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model.to(device)

# Loss functions
classification_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.SmoothL1Loss()

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001)
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

    return intersection / union if union > 0 else 0

# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    total_iou = 0
    num_samples = 0

    with torch.no_grad():
        for images, bboxes, labels in dataloader:
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

# Training loop with early stopping
early_stopping_patience = 3
early_stopping_counter = 0
best_accuracy = 0

# Change epoch nums
epochs = 100
for epoch in range(epochs):
    model.train()
    epoch_class_loss = 0
    epoch_bbox_loss = 0

    for images, bboxes, labels in dataloader:
        images, bboxes, labels = images.to(device), bboxes.to(device), labels.to(device)

        class_out, bbox_out = model(images)
        class_loss = classification_loss_fn(class_out, labels)
        bbox_loss = bbox_loss_fn(bbox_out, bboxes)
        total_loss = class_loss + bbox_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_class_loss += class_loss.item()
        epoch_bbox_loss += bbox_loss.item()

    classification_accuracy, mean_iou = evaluate_model(model, dataloader, device)
    print(f"Epoch {epoch + 1}/{epochs}, Classification Loss: {epoch_class_loss:.4f}, "
          f"BBox Loss: {epoch_bbox_loss:.4f}, Accuracy: {classification_accuracy:.2f}%, Mean IoU: {mean_iou:.4f}")

    scheduler.step(classification_accuracy)

    if classification_accuracy > best_accuracy:
        best_accuracy = classification_accuracy
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        early_stopping_counter += 1
        print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered.")
        break

model.load_state_dict(torch.load('best_model.pth'))
