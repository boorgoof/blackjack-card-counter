#Matteo Bino

from ultralytics import YOLO
import torch
import os

# Load standard model
model = YOLO('yolo11s.pt') 

if torch.cuda.is_available():
    device = 0  # Use the first GPU (index 0)
    print(f" GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("No GPU detected, using CPU.")

DATASET_NAME = "single_cards"

DATASET_PATH = f'./datasets/{DATASET_NAME}/data.yaml'

if not os.path.isfile(DATASET_PATH):
    print(f"{DATASET_PATH} does not exist!")
    exit()

results = model.train(
    data=DATASET_PATH,
    epochs=100,
    imgsz=1280,
    rect=True,
    batch=12,
    device=device,
    mosaic=1.0,
    degrees=15.0,
    fliplr=0.0,
    project=f"output/{DATASET_NAME}"
)
