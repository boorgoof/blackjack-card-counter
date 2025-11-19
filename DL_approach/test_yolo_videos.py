from ultralytics import YOLO
import torch


device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Running evaluation on: {device}")

model_path = 'runs/detect/train2/weights/best.pt' 


try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please verify that '{model_path}' exists.")
    exit()

# 3. Run Evaluation on the TEST split
print("Starting evaluation on the Test Set...")
metrics = model.val(
    data='datasets/video_test/data.yaml', # Path to your data config
    split='test',   # <--- This forces it to use the 'test' folder defined in yaml
    device=device,  # Run on GPU
    plots=True      # Generate confusion matrices and prediction samples
)

# 4. Print Results
print("\n--- Test Set Results ---")
print(f"mAP50 (Mean Average Precision @ IoU 0.5): {metrics.box.map50:.3f}")
print(f"mAP50-95 (Standard metric): {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")

print(f"\nDetailed results/plots saved to: {metrics.save_dir}")