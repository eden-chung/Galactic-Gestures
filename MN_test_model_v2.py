# test_model.py

import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Large_Weights
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

# Import the GestureDataset from your separate module
from gesture_dataset import GestureDataset

# Constants
INPUT_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5
MODEL_SAVE_PATH = 'V2_30epoch.pth'  # Ensure this matches the saved model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    criterion = nn.CrossEntropyLoss()

    # Initialize variables for collecting labels and predictions
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_losses = []
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    # Testing loop
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            all_losses.append(loss.item() * images.size(0))

            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Calculate overall test loss
    test_loss = sum(all_losses) / len(test_dataset)
    print(f"Test Loss: {test_loss:.4f}")

    # Calculate overall accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    class_names = [f"Class {i}" for i in range(NUM_CLASSES)]
    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))

    # Overall precision, recall, F1 score
    overall_precision = report['weighted avg']['precision']
    overall_recall = report['weighted avg']['recall']
    overall_f1 = report['weighted avg']['f1-score']

    print(f"Overall Precision: {overall_precision:.2f}")
    print(f"Overall Recall: {overall_recall:.2f}")
    print(f"Overall F1 Score: {overall_f1:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Normalized Confusion Matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.savefig('confusion_matrix_normalized.png')
    plt.close()

    # Precision, Recall, F1 Score per Class
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None, labels=range(NUM_CLASSES))

    # Plot Precision per Class
    x = np.arange(NUM_CLASSES)
    plt.figure()
    plt.bar(x, precision)
    plt.xticks(x, class_names)
    plt.xlabel('Class')
    plt.ylabel('Precision')
    plt.title('Precision per Class')
    plt.savefig('P_curve.png')
    plt.close()

    # Plot Recall per Class
    plt.figure()
    plt.bar(x, recall)
    plt.xticks(x, class_names)
    plt.xlabel('Class')
    plt.ylabel('Recall')
    plt.title('Recall per Class')
    plt.savefig('R_curve.png')
    plt.close()

    # Plot F1 Score per Class
    plt.figure()
    plt.bar(x, f1)
    plt.xticks(x, class_names)
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score per Class')
    plt.savefig('F1_curve.png')
    plt.close()

    # Plot Accuracy per Class
    class_accuracy = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(NUM_CLASSES)]
    plt.figure()
    plt.bar(x, class_accuracy)
    plt.xticks(x, class_names)
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Class')
    plt.savefig('per_class_accuracy.png')
    plt.close()

    # *** Added code to compare Class 0 and Class 4 ***

    # Plot comparison between Class 0 and Class 4 accuracy
    classes_to_compare = [0, 4]
    accuracy_to_compare = [class_accuracy[i] for i in classes_to_compare]
    class_names_to_compare = [class_names[i] for i in classes_to_compare]

    plt.figure()
    plt.bar(class_names_to_compare, accuracy_to_compare, color=['blue', 'green'])
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison: Class 0 vs Class 4')
    plt.ylim(0, 100)  # Ensure y-axis goes from 0% to 100%
    for i, acc in enumerate(accuracy_to_compare):
        plt.text(i, acc + 1, f'{acc:.2f}%', ha='center')
    plt.savefig('class_0_vs_4_accuracy.png')
    plt.close()

    print(f"Class {class_names_to_compare[0]} Accuracy: {accuracy_to_compare[0]:.2f}%")
    print(f"Class {class_names_to_compare[1]} Accuracy: {accuracy_to_compare[1]:.2f}%")
    if accuracy_to_compare[0] > accuracy_to_compare[1]:
        print(f"{class_names_to_compare[0]} has higher accuracy than {class_names_to_compare[1]}")
    elif accuracy_to_compare[0] < accuracy_to_compare[1]:
        print(f"{class_names_to_compare[1]} has higher accuracy than {class_names_to_compare[0]}")
    else:
        print(f"{class_names_to_compare[0]} and {class_names_to_compare[1]} have the same accuracy")

    # Binarize the labels for multi-label precision-recall curves
    y_test_binarized = label_binarize(all_labels, classes=range(NUM_CLASSES))
    y_score = np.array(all_probabilities)

    # Compute micro-average Precision-Recall curve and average precision
    precision_micro, recall_micro, _ = precision_recall_curve(y_test_binarized.ravel(), y_score.ravel())
    average_precision_micro = average_precision_score(y_test_binarized, y_score, average='micro')

    # Plot micro-average Precision-Recall curve
    plt.figure()
    plt.plot(recall_micro, precision_micro, label='Micro-average PR curve (AP={0:0.2f})'.format(average_precision_micro))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Micro-average Precision-Recall Curve')
    plt.legend()
    plt.savefig('PR_curve.png')
    plt.close()

    # Optionally, plot Precision-Recall curve for each class
    for i in range(NUM_CLASSES):
        precision_i, recall_i, _ = precision_recall_curve(y_test_binarized[:, i], y_score[:, i])
        average_precision_i = average_precision_score(y_test_binarized[:, i], y_score[:, i])
        plt.figure()
        plt.plot(recall_i, precision_i, label='AP={0:0.2f}'.format(average_precision_i))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {class_names[i]}')
        plt.legend()
        plt.savefig(f'PR_curve_class_{i}.png')
        plt.close()

    # Save sample images with predictions and labels
    def denormalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Function to save images for a batch
    def save_batch_images(images, labels, predictions, batch_idx, prefix):
        images = images.cpu()
        labels = labels.cpu()
        predictions = predictions.cpu()
        fig = plt.figure(figsize=(15, 10))
        for idx in range(min(BATCH_SIZE, 8)):
            ax = fig.add_subplot(2, 4, idx+1, xticks=[], yticks=[])
            img = denormalize(images[idx], mean, std)
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            if prefix == 'labels':
                ax.set_title(f'True: {class_names[labels[idx]]}')
            elif prefix == 'pred':
                ax.set_title(f'Predicted: {class_names[predictions[idx]]}')
        plt.tight_layout()
        plt.savefig(f'val_batch{batch_idx}_{prefix}.jpg')
        plt.close()

    # Get first two batches from test_loader
    dataiter = iter(test_loader)
    for batch_idx in range(2):  # Save two batches
        images, labels = next(dataiter)
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        # Save images with true labels
        save_batch_images(images, labels, predicted, batch_idx, 'labels')
        # Save images with predicted labels
        save_batch_images(images, labels, predicted, batch_idx, 'pred')

    # Plot Test Loss (since it's a single value, we can create a bar chart)
    plt.figure()
    plt.bar(['Test Loss'], [test_loss])
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.savefig('loss_curve.png')
    plt.close()

    # Plot Overall Accuracy, Precision, Recall, F1 Score
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [accuracy * 100, overall_precision * 100, overall_recall * 100, overall_f1 * 100]
    plt.figure()
    plt.bar(metrics_names, metrics_values)
    plt.ylabel('Percentage')
    plt.title('Overall Metrics')
    plt.savefig('overall_metrics.png')
    plt.close()

    print("Evaluation metrics and images have been saved.")