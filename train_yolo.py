import argparse
import sys
from pathlib import Path
from typing import Optional
import torch
from ultralytics import YOLO

DEFAULT_MODEL = "yolo11s.pt"
DEFAULT_EPOCHS = 50
DEFAULT_IMG_SIZE = 640
DEFAULT_BATCH_SIZE = 16
PROJECT_NAME = "output/yolo_train_results"
RUN_NAME = "yolo11s_cards"
DATASET_CONFIG_PATH = "data/datasets/YoloDataset/data.yaml"

def get_device(device_arg: Optional[str] = None) -> str:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def train_model(epochs: int, img_size: int, batch_size: int, device: str, dataset_path: Path):
    print(f"Device: {device}")
    
    try:
        model = YOLO(DEFAULT_MODEL)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    try:
        train_device = device if device != 'cpu' else None
        model.train(
            data=str(dataset_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            project=PROJECT_NAME,
            name=RUN_NAME,
            device=train_device
        )

        metrics = model.val()
        print(f"Validation Metrics: {metrics}")

        export_path = model.export(format="onnx")
        print(f"Model exported to: {export_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    
    args = parser.parse_args()

    dataset_path = Path(DATASET_CONFIG_PATH).resolve()
    if not dataset_path.exists():
        print(f"Config not found at {dataset_path}")
        sys.exit(1)

    final_epochs = 1 if args.smoke_test else args.epochs
    device = get_device(args.device)

    train_model(
        epochs=final_epochs,
        img_size=args.imgsz,
        batch_size=args.batch,
        device=device,
        dataset_path=dataset_path
    )

if __name__ == '__main__':
    main()
