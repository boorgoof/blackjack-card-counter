import numpy as np
import os
from ultralytics import YOLO
from pathlib import Path
import cv2 as cv


# --- CONFIGURATION ---
MODEL_PATH = 'models/yolov11s_single_cards_1280.pt'
DATA_YAML = 'datasets/single_cards/data.yaml'
IMG_SIZE = 1280
TARGET_W, TARGET_H = 1280, 960

OUTPUT_DIR = Path('output/evaluation/single_cards')
STATS_OUT = OUTPUT_DIR / "stats"
STATS_OUT.mkdir(parents=True, exist_ok=True)

IMG_OUT = OUTPUT_DIR / "annotated_images"
LBL_OUT = OUTPUT_DIR / "predicted_labels"

if(not os.path.exists(LBL_OUT)):
    os.makedirs(LBL_OUT)
if(not os.path.exists(IMG_OUT)):
    os.makedirs(IMG_OUT)
# Load model
model = YOLO(MODEL_PATH)

# 1. RUN VALIDATION (For Math Metrics)
print("Calculating metrics...")
metrics = model.val(data=DATA_YAML, imgsz=IMG_SIZE, split='val')

# 2. MEAN ACCURACY & mAP (Mean IoU equivalent in YOLO)
# YOLO uses mAP (mean Average Precision). 
# mAP50-95 is the standard "Accuracy/IoU" metric.
print(f"\n--- GLOBAL METRICS ---")
print(f"Mean Accuracy (mAP50): {metrics.box.map50:.4f}")
print(f"Mean IoU-based metric (mAP50-95): {metrics.box.map:.4f}")

# 3. PRECISION, RECALL, F1 FOR EACH CARD TYPE
print(f"\n--- CLASS-WISE METRICS ---")
print(f"{'Card Type':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 45)

# Get class names from model
names = model.names
# Precision, Recall, and F1 are arrays corresponding to the names
for i, name in names.items():
    p = metrics.box.class_result(i)[0] # Precision
    r = metrics.box.class_result(i)[1] # Recall
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    print(f"{name:<10} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")

# 4. CONFUSION MATRIX (Text format)
# This prints the raw matrix. Zeros mean no confusion, 
# numbers on the diagonal mean correct predictions.
print(f"\n--- CONFUSION MATRIX (RAW ARRAY) ---")
cm_array = metrics.confusion_matrix.matrix
print(cm_array.astype(int)) 

print("Saving Confusion Matrix...")
cm = metrics.confusion_matrix.matrix.astype(int)
np.savetxt(STATS_OUT / "confusion_matrix.txt", cm, fmt='%d', delimiter='\t')

print("Saving Metrics Report...")
with open(STATS_OUT / "metrics_report.txt", "w") as f:
    f.write("--- GLOBAL METRICS ---\n")
    f.write(f"Mean Accuracy (mAP50): {metrics.box.map50:.4f}\n")
    f.write(f"Mean IoU-based (mAP50-95): {metrics.box.map:.4f}\n\n")
    
    f.write("--- CLASS-WISE METRICS ---\n")
    header = f"{'Class_ID':<8} | {'Name':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}\n"
    f.write(header)
    f.write("-" * 55 + "\n")
    
    # metrics.box.p and metrics.box.r are arrays of precision/recall for each class
    for i in range(len(model.names)):
        p = metrics.box.class_result(i)[0]
        r = metrics.box.class_result(i)[1]
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        name = model.names[i]
        f.write(f"{i:<8} | {name:<8} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}\n")



# 5. PREDICTED ANNOTATIONS & IMAGES (With boxes)
print(f"\nGenerating visual predictions...")
results = model.predict(
    source='datasets/single_cards/images/val', 
    imgsz=IMG_SIZE, 
    name='predicted_images',
    stream=True
)
for r in results:
    img_name = Path(r.path).name
    lbl_name = Path(r.path).stem + ".txt"

    # A. Get annotated image (this is still at the model's internal scale or original)
    annotated_img = r.plot() 

    # B. Force Resize to 1280x960
    resized_img = cv.resize(annotated_img, (TARGET_W, TARGET_H), interpolation=cv.INTER_AREA)

    # C. Save Image
    cv.imwrite(str(IMG_OUT / img_name), resized_img)

    # D. Save Predicted Labels (YOLO format: class x_center y_center width height)
    
    with open(LBL_OUT / lbl_name, 'w') as f:
        for box in r.boxes:
            cls = int(box.cls[0])
            # xywhn provides normalized coordinates (0 to 1)
            coords = box.xywhn[0].tolist() 
            f.write(f"{cls} {' '.join(f'{c:.6f}' for c in coords)}\n")

print(f"\n--- EVALUATION COMPLETED ---")
print(f"Stats saved to: {STATS_OUT}")
print(f"Annotated Images (1280x960) saved to: {IMG_OUT}")
print(f"Predicted Labels saved to: {LBL_OUT}")