import json
import os
import shutil
import random
import yaml
from collections import defaultdict
from tqdm import tqdm

# Path to your single folder of images
DATASET_PATH = "../data/datasets/single_cards"
SOURCE_IMAGES_DIR = os.path.join(DATASET_PATH, "Images/Images") 
# Path to the JSON file you provided
JSON_FILE = os.path.join(DATASET_PATH, "annotation.json")

OUTPUT_DIR = "datasets/cards_stratified"

# Ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

def convert_coco_to_yolo_box(bbox, img_w, img_h):
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    return x_center, y_center, width, height

def main():
    # 1. Load JSON
    print(f"Loading {JSON_FILE}...")
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    # 2. Map Categories: JSON ID -> YOLO ID (0-indexed)
    # Sort by ID to ensure consistent order
    sorted_cats = sorted(data['categories'], key=lambda x: x['id'])
    class_names = [c['name'] for c in sorted_cats]
    json_id_to_yolo_id = {c['id']: i for i, c in enumerate(sorted_cats)}
    
    print(f"Classes detected: {len(class_names)}")

    # 3. Group Annotations by Image
    img_id_to_ann = {}
    for ann in data['annotations']:
        img_id_to_ann[ann['image_id']] = ann

    # 4. Group Images by Class (Stratification Preparation)
    # We create a dictionary: { yolo_class_id: [list_of_image_objects] }
    class_buckets = defaultdict(list)
    
    print("Grouping images by class...")
    for img_info in data['images']:
        img_id = img_info['id']
        
        # Skip if image has no annotation
        if img_id not in img_id_to_ann:
            continue

        ann = img_id_to_ann[img_id]
        json_cat_id = ann['category_id']
        yolo_class_id = json_id_to_yolo_id[json_cat_id]
        
        # Store the package needed to process later
        package = {
            'img_info': img_info,
            'ann': ann,
            'class_id': yolo_class_id
        }
        class_buckets[yolo_class_id].append(package)

    # 5. Perform Stratified Split
    final_splits = {'train': [], 'val': [], 'test': []}
    
    print("Performing stratified split...")
    random.seed(42) # Fix seed for reproducibility

    for class_id, items in class_buckets.items():
        random.shuffle(items)
        count = len(items)
        
        # Calculate cut points
        train_end = int(count * TRAIN_RATIO)
        val_end = int(count * (TRAIN_RATIO + VAL_RATIO))
        
        # Safety: Ensure at least 1 image goes to train if duplicates exist
        if count > 0 and train_end == 0:
            train_end = 1
        
        # Add to lists
        final_splits['train'].extend(items[:train_end])
        final_splits['val'].extend(items[train_end:val_end])
        final_splits['test'].extend(items[val_end:])

    # Print stats
    print(f"Total: {len(data['images'])}")
    print(f"Train: {len(final_splits['train'])}")
    print(f"Val:   {len(final_splits['val'])}")
    print(f"Test:  {len(final_splits['test'])}")

    # 6. Write Files
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for split_name, items in final_splits.items():
        # Create dirs
        img_dir = os.path.join(OUTPUT_DIR, split_name, 'images')
        lbl_dir = os.path.join(OUTPUT_DIR, split_name, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        print(f"Writing {split_name} data...")
        for item in tqdm(items):
            info = item['img_info']
            ann = item['ann']
            cls_id = item['class_id']
            
            # Source check
            src_path = os.path.join(SOURCE_IMAGES_DIR, info['file_name'])
            if not os.path.exists(src_path):
                continue
                
            # 1. Copy Image
            shutil.copy2(src_path, os.path.join(img_dir, info['file_name']))
            
            # 2. Create Label File
            # Convert Box
            xc, yc, w, h = convert_coco_to_yolo_box(ann['bbox'], info['width'], info['height'])
            
            txt_name = os.path.splitext(info['file_name'])[0] + ".txt"
            with open(os.path.join(lbl_dir, txt_name), 'w') as f:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # 7. Create data.yaml
    yaml_data = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
        
    print(f"Success! Stratified dataset created at {OUTPUT_DIR}")

if __name__ == "__main__":
    main()