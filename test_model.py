from ultralytics import YOLO
import cv2
import sys

def main():
    model_path = "output/yolo_train_results/yolo11s_cards3/weights/best.pt"
    image_path = "data/datasets/YoloDataset/valid/images/001934273_jpg.rf.bb75c1e46bcc02e67d1a6bedaf5ef623.jpg"
    
    print(f"Loading model from {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    print(f"Predicting on {image_path}")
    results = model(image_path)
    
    print(f"Detections: {len(results[0].boxes)}")
    if len(results[0].boxes) > 0:
        results[0].save(filename="test_prediction.jpg")
        print("Saved test_prediction.jpg")
    else:
        print("No detections found on validation image.")

if __name__ == "__main__":
    main()
