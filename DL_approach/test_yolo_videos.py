from ultralytics import YOLO
import torch


device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Running evaluation on: {device}")

model_path = './yolo11s_synthetic.pt'


try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please verify that '{model_path}' exists.")
    exit()

source_images = 'datasets/video_test/test/images'

# 3. Run Evaluation on the TEST split
print("Starting prediction on the Test Set...")
results = model.predict(
    source=source_images,
    conf=0.7,       # Confidence threshold (ignore weak detections)
    iou=0.5,         # NMS threshold (remove overlapping boxes)
    save=True,       # <--- This saves the images with drawings
    save_txt=False,  # Set True if you also want the coordinates in .txt files
    save_conf=True,  # Show confidence score on the image
    device=device,
    max_det=100,     # Maximum number of cards allowed per image
    
    # formatting options
    line_width=2,    # Thickness of the box lines
    show_labels=True # Show class names (e.g. "10S")
)
print(f"\nDone! Check your images in the folder: {results[0].save_dir}")
