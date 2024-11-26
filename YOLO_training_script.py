from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Configuration
data_yaml = 'data.yaml'
#model_weights = 'yolo_100_epochs.pt'
model_weights = 'yolo_300_epochs_best.pt'
img_size = 640
batch_size = 16
epochs = 300
early_stopping_patience = 10
project_dir = 'runs_large_300_epochs/train'
experiment_name = 'yolo'

# Load the pre trained YOLO model
model = YOLO(model_weights)
model.to(device)

# # Train with early stopping
# results = model.train(
#     data=data_yaml,
#     imgsz=img_size,
#     batch=batch_size,
#     epochs=epochs,
#     patience=early_stopping_patience,
#     project=project_dir,
#     name=experiment_name,
#     exist_ok=True,
#     augment=True,
#     lrf=0.01,
#     device = device,
# )

results = model.val(
    data=data_yaml,
    split='test', 
    imgsz=img_size,
    batch=batch_size,
    project='runs_large_300_epochs',
    name='test',
    device=device,
    exist_ok=True,
)

