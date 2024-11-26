from ultralytics import YOLO

# Configuration
data_yaml = 'data.yaml'
model_weights = 'yolov5s.pt'
img_size = 224
batch_size = 32
epochs = 100
early_stopping_patience = 10
project_dir = 'runs1/train'
experiment_name = 'yolo'

# Load the pre trained YOLO model
model = YOLO(model_weights)

# Train with early stopping
results = model.train(
    data=data_yaml,
    imgsz=img_size,
    batch=batch_size,
    epochs=epochs,
    patience=early_stopping_patience,
    project=project_dir,
    name=experiment_name,
    exist_ok=True
)

