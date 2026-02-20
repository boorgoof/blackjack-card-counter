import numpy as np
import os
from ultralytics import YOLO
from pathlib import Path
import cv2 as cv

# --- CONFIGURATION ---
MODEL_NAME = 'yolov11s_single_cards_1280'
MODEL_PATH = f'models/{MODEL_NAME}.pt'
DATA_YAML = 'datasets/single_cards/data.yaml'
GT_LABELS_DIR = Path('datasets/single_cards/labels/val') # Path to your Ground Truth
IMG_SIZE = 1280
TARGET_W, TARGET_H = 1280, 960

OUTPUT_DIR = Path(f'results/{MODEL_NAME}/test')
STATS_OUT = OUTPUT_DIR / "stats"
IMG_OUT = OUTPUT_DIR / "annotated_images"
LBL_OUT = OUTPUT_DIR / "predicted_labels"

for p in [STATS_OUT, IMG_OUT, LBL_OUT]:
    p.mkdir(parents=True, exist_ok=True)

def calculate_iou(box1, box2):
    """ Calculates IoU of two boxes in [x1, y1, x2, y2] format """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

# Load model
model = YOLO(MODEL_PATH)

# 1. RUN VALIDATION (For mAP50)
print("Calculating validation metrics...")
metrics = model.val(data=DATA_YAML, imgsz=IMG_SIZE, split='val')

# 2. GLOBAL STATS
mAP50 = metrics.box.map50  # mAP at IoU 0.5
cm_array = metrics.confusion_matrix.matrix
true_positives_count = np.trace(cm_array[:-1, :-1])
total_gt_count = np.sum(cm_array[:-1, :])
global_accuracy = true_positives_count / total_gt_count if total_gt_count > 0 else 0

# 3. PREDICT & COMPUTE MEAN IOU
print(f"Generating visual predictions and calculating Mean IoU...")
results = model.predict(source='datasets/single_cards/images/val', imgsz=IMG_SIZE, conf=0.25, stream=True)

all_ious = []

for r in results:
    img_name = Path(r.path).name
    lbl_name = Path(r.path).stem + ".txt"
    
    # Load Predicted Boxes [x1, y1, x2, y2]
    pred_boxes = r.boxes.xyxy.cpu().numpy()
    pred_classes = r.boxes.cls.cpu().numpy()

    # Load Ground Truth Boxes (if exists)
    gt_path = GT_LABELS_DIR / lbl_name
    if gt_path.exists():
        gt_data = np.loadtxt(gt_path).reshape(-1, 5) # class, x, y, w, h
        # Convert GT normalized xywh to absolute xyxy
        h, w = r.orig_shape
        gt_boxes_xyxy = []
        for row in gt_data:
            _, cx, cy, bw, bh = row
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h
            gt_boxes_xyxy.append([x1, y1, x2, y2])
        
        # Simple matching to find IoU for Mean IoU calculation
        for p_box, p_cls in zip(pred_boxes, pred_classes):
            best_iou = 0
            for i, (g_box) in enumerate(gt_boxes_xyxy):
                if p_cls == gt_data[i][0]: # Match only if same class
                    iou = calculate_iou(p_box, g_box)
                    best_iou = max(best_iou, iou)
            if best_iou > 0.1: # Only count if there's some overlap
                all_ious.append(best_iou)

    # Save Resized Annotated Image
    annotated_img = r.plot() 
    resized_img = cv.resize(annotated_img, (TARGET_W, TARGET_H), interpolation=cv.INTER_AREA)
    cv.imwrite(str(IMG_OUT / img_name), resized_img)

    # Save Predicted Labels
    with open(LBL_OUT / lbl_name, 'w') as f:
        for box in r.boxes:
            cls = int(box.cls[0])
            coords = box.xywhn[0].tolist() 
            f.write(f"{cls} {' '.join(f'{c:.6f}' for c in coords)}\n")

mean_iou = np.mean(all_ious) if all_ious else 0

# 4. FINAL REPORTING
print(f"\n--- EVALUATION RESULTS ---")
print(f"Global Accuracy:  {global_accuracy:.4f}")
print(f"mAP50:  {mAP50:.4f}")
print(f"Mean IoU:  {mean_iou:.4f}")

with open(STATS_OUT / "metrics_report.txt", "w") as f:
    f.write("--- SUMMARY METRICS ---\n")
    f.write(f"Global Accuracy:  {global_accuracy:.4f}\n")
    f.write(f"mAP50:  {mAP50:.4f}\n")
    f.write(f"Mean IoU:  {mean_iou:.4f}\n\n")
    
    f.write("--- CLASS-WISE METRICS ---\n")
    f.write(f"{'Class_ID':<8} | {'Name':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}\n")
    f.write("-" * 55 + "\n")
    for i in range(len(model.names)):
        p, r_val = metrics.box.class_result(i)[0], metrics.box.class_result(i)[1]
        f1 = 2 * (p * r_val) / (p + r_val) if (p + r_val) > 0 else 0
        f.write(f"{i:<8} | {model.names[i]:<8} | {p:<10.4f} | {r_val:<10.4f} | {f1:<10.4f}\n")

np.savetxt(STATS_OUT / "confusion_matrix.txt", cm_array.astype(int), fmt='%d', delimiter='\t')

print(f"\nStats saved to: {STATS_OUT}")
print(f"Images (1280x960) saved to: {IMG_OUT}")