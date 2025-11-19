import json
import os
import shutil
import random
import yaml
from tqdm import tqdm  # Optional: pip install tqdm for progress bars

# --- CONFIGURATION ---
# Path to your single folder of images
DATASET_PATH = "../data/datasets/single_cards"
SOURCE_IMAGES_DIR = os.path.join(DATASET_PATH, "Images/Images") 
# Path to the JSON file you provided
JSON_FILE = os.path.join(DATASET_PATH, "annotation.json")

# Where to create the YOLO dataset
OUTPUT_DIR = "datasets/cards"

# Split Ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
# Test gets the remainder (0.1)

random.seed(42)

def convert_coco_to_yolo_box(bbox, img_w, img_h):
    """
    Converts COCO bbox [x_min, y_min, width, height] 
    to YOLO bbox [x_center, y_center, width, height] normalized (0-1).
    """
    x_min, y_min, w, h = bbox
    
    # Calculate center coordinates
    x_center = x_min + (w / 2)
    y_center = y_min + (h / 2)
    
    # Normalize by image dimensions
    x_center /= img_w
    y_center /= img_h
    w /= img_w
    h /= img_h
    
    return x_center, y_center, w, h

def main():
    # 1. Load the JSON data
    print(f"Loading {JSON_FILE}...")
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    # 2. Process Categories (Classes)
    # We need to map JSON category_id to YOLO class_id (0-indexed)
    # Based on your file, IDs start at 1 ("AS"). YOLO needs 0.
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Sort by ID to ensure deterministic order
    sorted_ids = sorted(categories.keys())
    
    # Create a list of names for the YAML file
    class_names = [categories[i] for i in sorted_ids]
    
    # Map original ID -> New YOLO ID (0 to 52)
    id_map = {original_id: idx for idx, original_id in enumerate(sorted_ids)}
    
    print(f"Found {len(class_names)} classes: {class_names[:5]}...{class_names[-1]}")

    # 3. Organize Annotations by Image ID
    print("Grouping annotations...")
    img_to_anns = {img['id']: [] for img in data['images']}
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    # 4. Prepare Image Data List
    image_entries = []
    for img_info in data['images']:
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        # Check if image file actually exists
        src_path = os.path.join(SOURCE_IMAGES_DIR, file_name)
        if not os.path.exists(src_path):
            print(f"Warning: Image {file_name} found in JSON but missing in folder. Skipping.")
            continue
            
        anns = img_to_anns.get(img_id, [])
        
        # Store everything needed to process this image later
        image_entries.append({
            'info': img_info,
            'anns': anns,
            'src_path': src_path
        })

    # 5. Shuffle and Split
    
    random.shuffle(image_entries)
    
    total = len(image_entries)
    train_end = int(total * TRAIN_RATIO)
    val_end = int(total * (TRAIN_RATIO + VAL_RATIO))
    
    splits = {
        'train': image_entries[:train_end],
        'val': image_entries[train_end:val_end],
        'test': image_entries[val_end:]
    }
    
    print(f"Split results: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    # 6. Process and Move Files
    # Re-create output directories
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    for split_name, entries in splits.items():
        img_dir = os.path.join(OUTPUT_DIR, split_name, 'images')
        lbl_dir = os.path.join(OUTPUT_DIR, split_name, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        print(f"Processing {split_name} set...")
        
        for entry in tqdm(entries):
            img_info = entry['info']
            anns = entry['anns']
            
            # Copy Image
            shutil.copy2(entry['src_path'], os.path.join(img_dir, img_info['file_name']))
            
            # Generate Label File
            txt_filename = os.path.splitext(img_info['file_name'])[0] + ".txt"
            txt_path = os.path.join(lbl_dir, txt_filename)
            
            with open(txt_path, 'w') as f_txt:
                for ann in anns:
                    # Convert Category ID
                    cat_id_json = ann['category_id']
                    class_id = id_map[cat_id_json]
                    
                    # Convert Coordinates
                    # JSON bbox is [x, y, w, h] absolute
                    bbox = ann['bbox']
                    xc, yc, w, h = convert_coco_to_yolo_box(bbox, img_info['width'], img_info['height'])
                    
                    # Write YOLO line: class_id x_center y_center width height
                    f_txt.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # 7. Generate data.yaml
    print("Generating data.yaml...")
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"Done! Dataset prepared at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()