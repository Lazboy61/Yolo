import os
import shutil
import random

# Locaties
SOURCE_IMAGE_DIR = r"D:\Users\Gebruiker\Documents\Project D\Yolo\reduced_dataset"
SOURCE_LABEL_DIR = os.path.join(SOURCE_IMAGE_DIR, "labels")

DEST_IMAGE_DIR = r"D:\Users\Gebruiker\Documents\Project D\Yolo\dataset\images"
DEST_LABEL_DIR = r"D:\Users\Gebruiker\Documents\Project D\Yolo\dataset\labels"

# Maak output folders aan
for split in ["train", "val"]:
    os.makedirs(os.path.join(DEST_IMAGE_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(DEST_LABEL_DIR, split), exist_ok=True)

# Verwerk alle lettermappen (A-Z, etc.)
all_image_paths = []
for letter in os.listdir(SOURCE_IMAGE_DIR):
    image_folder = os.path.join(SOURCE_IMAGE_DIR, letter)
    label_folder = os.path.join(SOURCE_LABEL_DIR, letter)
    
    if os.path.isdir(image_folder) and os.path.isdir(label_folder):
        for file in os.listdir(image_folder):
            if file.endswith(".jpg"):
                img_path = os.path.join(image_folder, file)
                label_path = os.path.join(label_folder, file.replace(".jpg", ".txt"))
                if os.path.exists(label_path):
                    all_image_paths.append((img_path, label_path))

# Shuffle en split in train/val
random.shuffle(all_image_paths)
split_idx = int(len(all_image_paths) * 0.8)
train_set = all_image_paths[:split_idx]
val_set = all_image_paths[split_idx:]

def copy_pairs(pairs, dest_image_subdir, dest_label_subdir):
    for img_path, label_path in pairs:
        basename = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(dest_image_subdir, basename))
        shutil.copy(label_path, os.path.join(dest_label_subdir, basename.replace(".jpg", ".txt")))

copy_pairs(train_set, os.path.join(DEST_IMAGE_DIR, "train"), os.path.join(DEST_LABEL_DIR, "train"))
copy_pairs(val_set, os.path.join(DEST_IMAGE_DIR, "val"), os.path.join(DEST_LABEL_DIR, "val"))

print(f"✅ Gekopieerd: {len(train_set)} train + {len(val_set)} val")
