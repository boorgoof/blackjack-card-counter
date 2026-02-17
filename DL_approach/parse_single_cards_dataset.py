import os
import shutil
import random
from pathlib import Path

# --- CONFIGURATION ---
SOURCE_DIR = Path("../data/datasets/single_cards")
IMG_DIR = SOURCE_DIR / "Images" / "Images"
LBL_DIR = SOURCE_DIR / "YOLO_Annotations" / "YOLO_Annotations"

# Target structure
TARGET_DIR = Path("datasets/single_cards")
TRAIN_RATIO = 0.8

# The mapping provided by the user
# Note: In your filenames, cards are likely '2C', '10H', but the table uses 'C2', 'H10'.
# We will create a helper to map the filename prefix to the table index.
LABEL_NAMES = {
    0: "C10", 1: "C2", 2: "C3", 3: "C4", 4: "C5", 5: "C6", 6: "C7", 7: "C8", 8: "C9", 9: "CA", 10: "CJ", 11: "CK", 12: "CQ",
    13: "D10", 14: "D2", 15: "D3", 16: "D4", 17: "D5", 18: "D6", 19: "D7", 20: "D8", 21: "D9", 22: "DA", 23: "DJ", 24: "DK", 25: "DQ",
    26: "H10", 27: "H2", 28: "H3", 29: "H4", 30: "H5", 31: "H6", 32: "H7", 33: "H8", 34: "H9", 35: "HA", 36: "HJ", 37: "HK", 38: "HQ",
    39: "S10", 40: "S2", 41: "S3", 42: "S4", 43: "S5", 44: "S6", 45: "S7", 46: "S8", 47: "S9", 48: "SA", 49: "SJ", 50: "SK", 51: "SQ"
}

# Inverse mapping to find ID by Suit+Rank (e.g., "C2" -> 1)
NAME_TO_ID = {v: k for k, v in LABEL_NAMES.items()}

def get_card_type(filename):
    """
    Extracts the card type from filenames like '2C0.jpg' or '10H15.jpg'.
    Returns the key compatible with NAME_TO_ID (e.g., 'C2', 'H10').
    """
    stem = Path(filename).stem # '2C0'
    # Filenames are: [Rank][Suit][ID]
    # Rank can be 1 or 2 digits/chars (2-9, 10, A, J, Q, K)
    # Suit is 1 char (C, D, H, S)
    
    # Check for JOKER
    if stem.startswith("JOKER"):
        return None

    if stem.startswith('10'):
        rank = '10'
        suit = stem[2]
    else:
        rank = stem[0]
        suit = stem[1]
    
    # User table format is [Suit][Rank], so '2C' becomes 'C2'
    return f"{suit}{rank}"

def setup_dirs():
    for split in ['train', 'val']:
        (TARGET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def process_label(src_path, dst_path, class_id):
    """Reads the old label, forces the correct class ID from the table, and saves."""
    if not src_path.exists():
        return False
    
    with open(src_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0:
            # Replace the original class ID with our mapped class_id
            parts[0] = str(class_id)
            new_lines.append(" ".join(parts))
            
    with open(dst_path, 'w') as f:
        f.write("\n".join(new_lines))
    return True

def main():
    setup_dirs()
    
    # 1. Group images by card type
    all_images = list(IMG_DIR.glob("*.jpg"))
    groups = {}
    
    for img_path in all_images:
        card_type = get_card_type(img_path.name)
        if card_type not in groups:
            groups[card_type] = []
        groups[card_type].append(img_path)
    
    print(f"Found {len(groups)} card types.")

    # 2. Split each group and move files
    for card_type, images in groups.items():
        if card_type not in NAME_TO_ID:
            print(f"Warning: Card type {card_type} not found in mapping table. Skipping.")
            continue
        
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        print(f"{card_type}")
        class_id = NAME_TO_ID[card_type]
        
        for split, image_list in [('train', train_images), ('val', val_images)]:
            for img_path in image_list:
                # File names
                img_name = img_path.name
                lbl_name = img_path.stem + ".txt"
                
                # Destination paths
                dst_img = TARGET_DIR / "images" / split / img_name
                dst_lbl = TARGET_DIR / "labels" / split / lbl_name
                
                # Copy Image
                shutil.copy(img_path, dst_img)
                
                # Process and Move Label (ensuring ID matches table)
                src_lbl = LBL_DIR / lbl_name
                if not process_label(src_lbl, dst_lbl, class_id):
                    print(f"Warning: Label not found for {img_name}")

    # 3. Create data.yaml
    yaml_content = f"""
train: {TARGET_DIR}/images/train
val: {TARGET_DIR}/images/val

nc: 52
names:
"""
    for i in range(52):
        yaml_content += f"  {i}: {LABEL_NAMES[i]}\n"
        
    with open(TARGET_DIR / "data.yaml", "w") as f:
        f.write(yaml_content.strip())

    print("Success! Dataset split and data.yaml generated.")

if __name__ == "__main__":
    main()