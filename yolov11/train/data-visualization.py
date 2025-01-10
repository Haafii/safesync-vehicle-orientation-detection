import os
import matplotlib.pyplot as plt
from collections import Counter

# Directories for labels
train_labels_dir = "data/train/labels"
val_labels_dir = "data/val/labels"

# Class names corresponding to class IDs
class_names = [
    "car_back", "car_side", "car_front",
    "bus_back", "bus_side", "bus_front",
    "truck_back", "truck_side", "truck_front",
    "motorcycle_back", "motorcycle_side", "motorcycle_front",
    "bicycle_back", "bicycle_side", "bicycle_front"
]

# Function to count class occurrences
def count_classes(labels_dir):
    class_counts = Counter()
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    total_files = len(label_files)
    print(f"Counting classes in {labels_dir} (Total files: {total_files})")
    
    if total_files == 0:
        print(f"Warning: No label files found in {labels_dir}")
    
    for idx, label_file in enumerate(label_files, 1):
        if idx % 10 == 0:  # Print progress every 10 files
            print(f"Processing file {idx}/{total_files}...")
        with open(os.path.join(labels_dir, label_file), 'r') as file:
            for line in file:
                class_id = int(line.split()[0])  # Extract class ID
                class_counts[class_id] += 1
    print(f"Finished counting classes in {labels_dir}.")
    return class_counts

# Count classes in training and validation datasets
print("Counting classes in training dataset...")
train_class_counts = count_classes(train_labels_dir)
print("Counting classes in validation dataset...")
val_class_counts = count_classes(val_labels_dir)

# Combine counts into dictionaries for plotting
train_counts = {class_names[i]: train_class_counts.get(i, 0) for i in range(len(class_names))}
val_counts = {class_names[i]: val_class_counts.get(i, 0) for i in range(len(class_names))}
total_counts = {cls: train_counts[cls] + val_counts[cls] for cls in class_names}

# Create a folder for saving charts
output_dir = "charts"
os.makedirs(output_dir, exist_ok=True)

# Function to plot and save a bar chart
def plot_and_save(counts, title, filename):
    print(f"Generating chart for {title}...")
    plt.figure(figsize=(12, 8))
    plt.bar(counts.keys(), counts.values(), color='skyblue')
    plt.title(title, fontsize=16)
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Number of Instances', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Display the counts on top of the bars
    for i, count in enumerate(counts.values()):
        plt.text(i, count + 1, str(count), ha='center', fontsize=10)

    # Save the plot
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {title} as {save_path}")

# Plot and save the charts
print("Starting to generate the charts...")
plot_and_save(total_counts, "Class Distribution - Whole Dataset", "whole_dataset.png")
plot_and_save(train_counts, "Class Distribution - Training Dataset", "train_dataset.png")
plot_and_save(val_counts, "Class Distribution - Validation Dataset", "val_dataset.png")

print("All charts have been saved successfully!")
