import os
import shutil
import random

# Define the proportions for splitting the data
train_split = 0.8  # 80% for training
val_split = 0.2    # 20% for validation

# Create the 'data' folder structure
data_dir = "data"
train_images_dir = os.path.join(data_dir, "train/images")
train_labels_dir = os.path.join(data_dir, "train/labels")
val_images_dir = os.path.join(data_dir, "val/images")
val_labels_dir = os.path.join(data_dir, "val/labels")

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

print("Folder structure created successfully.")

# Get the list of all .jpg and .txt files
files = [f for f in os.listdir('.') if os.path.isfile(f)]
image_files = [f for f in files if f.endswith('.jpg')]
label_files = [f for f in files if f.endswith('.txt')]

# Ensure each image has a corresponding label file
paired_files = []
for img in image_files:
    label = os.path.splitext(img)[0] + ".txt"
    if label in label_files:
        paired_files.append((img, label))

print(f"Found {len(paired_files)} pairs of images and labels.")

# Shuffle and split data
random.shuffle(paired_files)
split_idx = int(len(paired_files) * train_split)
train_files = paired_files[:split_idx]
val_files = paired_files[split_idx:]

# Copy files to their respective directories
for img, label in train_files:
    train_img_path = os.path.join(train_images_dir, img)
    train_label_path = os.path.join(train_labels_dir, label)
    
    shutil.copy(img, train_img_path)
    shutil.copy(label, train_label_path)
    
    # Check if both the image and label were copied
    if os.path.exists(train_img_path) and os.path.exists(train_label_path):
        print(f"Copied {img} and {label} to training directory.")
    else:
        print(f"Error copying {img} or {label} to training directory.")

for img, label in val_files:
    val_img_path = os.path.join(val_images_dir, img)
    val_label_path = os.path.join(val_labels_dir, label)
    
    shutil.copy(img, val_img_path)
    shutil.copy(label, val_label_path)
    
    # Check if both the image and label were copied
    if os.path.exists(val_img_path) and os.path.exists(val_label_path):
        print(f"Copied {img} and {label} to validation directory.")
    else:
        print(f"Error copying {img} or {label} to validation directory.")

print("Dataset split and files copied successfully!")
