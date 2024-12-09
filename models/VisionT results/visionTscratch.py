import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as T
from torchvision.transforms import RandAugment

# Load dataset
df = pd.read_csv('annotations.csv')

# Visualization Function (Optional)
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

# Uncomment below to visualize some examples
# for i in range(3):
#     visualize_image(df.iloc[i])

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
        bbox = torch.tensor([row['x_start'], row['y_start'], row['x_end'], row['y_end']], dtype=torch.float32)
        label = torch.tensor(row['class_label'], dtype=torch.long)

        img = Image.open(img_path).convert("RGB")
        original_size = img.size
        width_ratio = self.target_size[0] / original_size[0]
        height_ratio = self.target_size[1] / original_size[1]
        bbox = bbox * torch.tensor([width_ratio, height_ratio, width_ratio, height_ratio])

        if self.transform:
            img = self.transform(img)

        return img, bbox, label

# Retrieve divese data augmentation
transform = T.Compose([
    T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    RandAugment(),  
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoader
dataset = VisionDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Vision Transformer from Scratch
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size

        # Layers
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.class_head = nn.Linear(dim, num_classes)
        self.bbox_head = nn.Linear(dim, 4)

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embedding(x)  # Shape: [batch_size, dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # Shape: [batch_size, num_patches, dim]
        b, n, _ = x.shape

        # Add CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [batch_size, num_patches + 1, dim]

        # Add positional encoding
        x = x + self.pos_embedding[:, :n + 1, :]

        # Transformer layers
        x = self.transformer(x)

        # Classification and BBox outputs
        cls_output = x[:, 0]  # Take CLS token output
        class_out = self.class_head(cls_output)
        bbox_out = self.bbox_head(cls_output)

        return class_out, bbox_out

# Initialize Model
num_classes = df['class_label'].nunique()
model = VisionTransformer(image_size=224, patch_size=16, num_classes=num_classes, dim=512, depth=6, heads=8, mlp_dim=2048)


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Loss Functions and Optimizer
classification_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# IoU Calculation
def calculate_iou(pred_box, gt_box):
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[2], gt_box[2])
    y2 = min(pred_box[3], gt_box[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = max(0, pred_box[2] - pred_box[0]) * max(0, pred_box[3] - pred_box[1])
    gt_area = max(0, gt_box[2] - gt_box[0]) * max(0, gt_box[3] - gt_box[1])
    union = pred_area + gt_area - intersection

    return intersection / union if union > 0 else 0

# Evaluation Function
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

            # Classification accuracy
            _, predicted = torch.max(class_out, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # IoU
            for pred_box, gt_box in zip(bbox_out, bboxes):
                iou = calculate_iou(pred_box.cpu().numpy(), gt_box.cpu().numpy())
                total_iou += iou
                num_samples += 1

    classification_accuracy = 100 * correct / total
    mean_iou = total_iou / num_samples if num_samples > 0 else 0

    return classification_accuracy, mean_iou

# Training Loop with Early Stopping
early_stopping_patience = 5
early_stopping_counter = 0
best_accuracy = 0

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
        total_loss = 0.7 * class_loss + 0.3 * bbox_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_class_loss += class_loss.item()
        epoch_bbox_loss += bbox_loss.item()

    classification_accuracy, mean_iou = evaluate_model(model, dataloader, device)
    print(f"Epoch {epoch + 1}/{epochs}, Classification Loss: {epoch_class_loss:.4f}, "
          f"BBox Loss: {epoch_bbox_loss:.4f}, Accuracy: {classification_accuracy:.2f}%, Mean IoU: {mean_iou:.4f}")
    scheduler.step(classification_accuracy)

    # Early Stopping Logic
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

# Load the best model after training
model.load_state_dict(torch.load('best_model.pth'))
