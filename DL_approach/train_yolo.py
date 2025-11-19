from ultralytics import YOLO
import torch

# Load standard model
model = YOLO('yolov8n.pt') 

if torch.cuda.is_available():
    device = 0  # Use the first GPU (index 0)
    print(f" GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("No GPU detected, using CPU (this will be slow!)")


# Train using the new Stratified data.yaml
results = model.train(
    data='datasets/cards_stratified/data.yaml', 
    epochs=50, 
    imgsz=640,
    device=device,
    mosaic=1.0,
    degrees=10.0,
    fliplr=0.0
)