import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

# Config
input_img_dir = "images\\train"
input_label_dir = "labels\\train"
output_img_dir = "images\\train_aug"
output_label_dir = "labels\\train_aug"
augment_factor = 3  # Number of copies per image

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

for img_file in tqdm(os.listdir(input_img_dir)):
    if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
        
    # Load image
    img_path = os.path.join(input_img_dir, img_file)
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    # Load corresponding label
    label_file = os.path.splitext(img_file)[0] + '.txt'
    label_path = os.path.join(input_label_dir, label_file)
    if not os.path.exists(label_path):
        continue
        
    # Original copy (Windows-compatible file operations)
    shutil.copy(img_path, os.path.join(output_img_dir, img_file))
    shutil.copy(label_path, os.path.join(output_label_dir, label_file))
    
    # Create augmented versions
    for i in range(augment_factor):
        # Random augmentation
        angle = np.random.randint(-15, 15)
        flip = np.random.choice([True, False])
        
        # Rotate
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w,h))
        
        # Flip
        if flip:
            rotated = cv2.flip(rotated, 1)
        
        # Save augmented image
        new_img_file = f"{os.path.splitext(img_file)[0]}_aug{i}.jpg"
        cv2.imwrite(os.path.join(output_img_dir, new_img_file), rotated)
        
        # Copy labels (use shutil instead of cp)
        new_label_file = f"{os.path.splitext(img_file)[0]}_aug{i}.txt"
        shutil.copy(label_path, os.path.join(output_label_dir, new_label_file))

print(f"Dataset augmented! New images in {output_img_dir}")