import json
import os
import shutil
from ultralytics import YOLO
from tqdm import tqdm

# --- CONFIGURATION ---
# Path to your ALREADY TRAINED model (to get its class list)
TRAINED_MODEL_PATH = 'runs/detect/train2/weights/best.pt'

# Paths to the NEW dataset
DATASET_PATH = "../data/datasets/videos"
NEW_DATASET_IMAGES = os.path.join(DATASET_PATH, "Images/Images")
NEW_DATASET_LABELS = os.path.join(DATASET_PATH, "YOLO_Annotations/YOLO_Annotations")
NEW_DATASET_JSON = os.path.join(DATASET_PATH, "notes.json") # Contains the category list

# Where to save the converted dataset
OUTPUT_DIR = "datasets/video_test"

def normalize_name(name):
    """
    Converts '10 Spades' -> '10S' and fixes typos.
    """
    # 1. Fix known typos in notes.json
    if "Hearth" in name:
        name = name.replace("Hearth", "Hearts")

    # 2. Split "Rank Suit" (e.g., "10 Spades")
    parts = name.split()
    
    if len(parts) == 2:
        rank = parts[0]      # "10"
        suit = parts[1]      # "Spades"
        suit_code = suit[0].upper() # "S"
        
        # Combine to standard code: "10S"
        return f"{rank}{suit_code}"
        
    # Fallback for unexpected formats
    return name.replace(" ", "").upper()

def main():
    # 1. Load Trained Model Class Names
    print(f"Loading model from {TRAINED_MODEL_PATH}...")
    model = YOLO(TRAINED_MODEL_PATH)
    model_classes = model.names # {0: 'AS', 1: '2S', ...}
    
    # Create reverse map: Name -> Model_ID
    # e.g. {'AS': 0, '2S': 1}
    model_name_to_id = {normalize_name(v): k for k, v in model_classes.items()}
    
    print(f"Model knows {len(model_classes)} classes.")

    # 2. Load New Dataset Class Names
    print(f"Loading {NEW_DATASET_JSON}...")
    with open(NEW_DATASET_JSON, 'r') as f:
        new_data = json.load(f)
    
    # Assuming json structure has 'categories': [{'id': 0, 'name': '...'}, ...]
    # If structure is different, adjust this line.
    new_categories = new_data.get('categories', [])
    
    # Create map: New_ID -> Model_ID
    id_translation_map = {}
    
    print("Mapping classes...")
    matched_count = 0
    for cat in new_categories:
        new_id = cat['id']
        raw_name = cat['name']
        norm_name = normalize_name(raw_name)
        
        if norm_name in model_name_to_id:
            model_id = model_name_to_id[norm_name]
            id_translation_map[new_id] = model_id
            matched_count += 1
        else:
            print(f"Warning: Class '{raw_name}' in new dataset not found in trained model.")

    print(f"Mapped {matched_count}/{len(new_categories)} categories.")

    # 3. Create Output Directories
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    # We treat this whole folder as a 'test' split
    img_dest = os.path.join(OUTPUT_DIR, 'test', 'images')
    lbl_dest = os.path.join(OUTPUT_DIR, 'test', 'labels')
    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(lbl_dest, exist_ok=True)

    # 4. Process Files
    print("Converting annotations and copying images...")
    
    # List all txt files
    txt_files = [f for f in os.listdir(NEW_DATASET_LABELS) if f.endswith('.txt')]
    
    skipped_imgs = 0
    
    for txt_file in tqdm(txt_files):
        # Define paths
        src_txt_path = os.path.join(NEW_DATASET_LABELS, txt_file)
        
        # Image extensions to try (since we don't know if it's jpg or png)
        image_name = os.path.splitext(txt_file)[0]
        found_img_path = None
        found_img_name = None
        
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            temp_path = os.path.join(NEW_DATASET_IMAGES, image_name + ext)
            if os.path.exists(temp_path):
                found_img_path = temp_path
                found_img_name = image_name + ext
                break
        
        if not found_img_path:
            # Image doesn't exist for this label
            skipped_imgs += 1
            continue

        # Read and Convert Label
        new_lines = []
        with open(src_txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5: continue
                
                current_id = int(parts[0])
                coords = parts[1:] # x, y, w, h
                
                if current_id in id_translation_map:
                    target_id = id_translation_map[current_id]
                    new_lines.append(f"{target_id} {' '.join(coords)}\n")
        
        # Only save if we have valid lines
        if new_lines:
            # Copy Image
            shutil.copy2(found_img_path, os.path.join(img_dest, found_img_name))
            
            # Write new Label
            with open(os.path.join(lbl_dest, txt_file), 'w') as f_out:
                f_out.writelines(new_lines)

    print(f"Processing complete. Skipped {skipped_imgs} missing images.")

    # 5. Generate YAML for testing
    yaml_data = {
        'path': os.path.abspath(OUTPUT_DIR),
        'test': 'test/images', # Point to the folder we just made
        'nc': len(model_classes),
        'names': model_classes
    }
    
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        # Simple string dump to avoid yaml dependency if not desired, 
        # but typically yaml library is installed with ultralytics
        import yaml
        yaml.dump(yaml_data, f, sort_keys=False)
    
    print(f"Ready for testing! Config saved at: {yaml_path}")

if __name__ == "__main__":
    main()