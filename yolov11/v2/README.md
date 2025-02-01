# Vehicle Orientation Detection with YOLOv11n

This repository contains the implementation and results of a vehicle orientation detection system using the YOLOv11n architecture. The project utilizes a comprehensive dataset and provides detailed visualizations of the model's performance.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Class Distribution](#class-distribution)
4. [Model Performance](#model-performance)
5. [Visualizations](#visualizations)

## Introduction

Vehicle orientation detection is essential for applications such as autonomous driving and wrong side detection. This project leverages the YOLOv11n model for accurate and efficient vehicle orientation classification.

## Dataset

The dataset consists of labeled images for various vehicle orientations. The classes include:

- `car_back`
- `car_side`
- `car_front`
- `bus_back`
- `bus_side`
- `bus_front`
- `truck_back`
- `truck_side`
- `truck_front`
- `motorcycle_back`
- `motorcycle_side`
- `motorcycle_front`
- `bicycle_back`
- `bicycle_side`
- `bicycle_front`

## Class Distribution

Understanding the class distribution is crucial for evaluating the dataset's balance and the model's performance.

- **Training Dataset**:
  ![Class Distribution - Training Dataset](https://github.com/Haafii/safesync-vehicle-orientation-detection/blob/main/yolov11/v2/charts/train_dataset.png)
  *Explanation*: The training dataset shows a high number of instances for `car_back` and `car_front`, indicating a class imbalance that the model needs to handle.
- **Validation Dataset**:
  ![Class Distribution - Validation Dataset](https://github.com/Haafii/safesync-vehicle-orientation-detection/blob/main/yolov11/v2/charts/val_dataset.png)
  *Explanation*: The validation dataset has a similar distribution to the training set, ensuring that the model is evaluated on a representative sample.
- **Whole Dataset**:
  ![Class Distribution - Whole Dataset](https://github.com/Haafii/safesync-vehicle-orientation-detection/blob/main/yolov11/v2/charts/whole_dataset.png) ![Class Distribution - Whole Dataset](https://github.com/Haafii/safesync-vehicle-orientation-detection/blob/main/yolov11/v2/charts/whole_dataset.png)
  *Explanation*: The whole dataset distribution provides an overview of the data distribution across all classes, helping in understanding the overall class balance.

## Model Performance

The model's performance is evaluated using various metrics and visualizations.

- **Normalized Confusion Matrix**:
  ![Confusion Matrix Normalized](https://raw.githubusercontent.com/your-username/repo-name/main/confusion_matrix_normalized.png)
  *Explanation*: The normalized confusion matrix provides a detailed view of the model's performance across classes, highlighting both correct predictions and misclassifications.
- **Raw Confusion Matrix**:
  ![Confusion Matrix](https://raw.githubusercontent.com/your-username/repo-name/main/confusion_matrix.png)
  *Explanation*: The raw confusion matrix shows the absolute numbers of correct and incorrect predictions, offering a clear view of the model's accuracy.
- **F1-Confidence Curve**:
  ![F1-Confidence Curve](https://raw.githubusercontent.com/your-username/repo-name/main/f1_confidence_curve.png)
  *Explanation*: The F1-confidence curve demonstrates the model's F1 score across different confidence thresholds, indicating the model's reliability at various confidence levels.
- **Precision-Confidence Curve**:
  ![Precision-Confidence Curve](https://raw.githubusercontent.com/your-username/repo-name/main/precision_confidence_curve.png)
  *Explanation*: This curve shows how precision varies with confidence, helping to understand the model's behavior under different confidence settings.
- **Precision-Recall Curve**:
  ![Precision-Recall Curve](https://raw.githubusercontent.com/your-username/repo-name/main/precision_recall_curve.png)
  *Explanation*: The precision-recall curve provides insights into the model's performance in terms of precision and recall, crucial for understanding the model's effectiveness.
- **Recall-Confidence Curve**:
  ![Recall-Confidence Curve](https://raw.githubusercontent.com/your-username/repo-name/main/recall_confidence_curve.png)
  *Explanation*: The recall-confidence curve illustrates how recall changes with confidence, offering insights into the model's ability to identify all instances of a class.

### Visualizations

Additional visualizations provide further insights into the model's behavior and the dataset characteristics.

![X and Y Coordinates](https://github.com/Haafii/safesync-vehicle-orientation-detection/blob/main/yolov11/v2/runs/detect/train/labels_correlogram.jpg)

- **X and Y Coordinates**:
  *Explanation*: The distribution of X and Y coordinates helps in understanding the spatial distribution of objects within the images.
- **Width and Height**:
  *Explanation*: The width and height distribution provides insights into the size variability of objects in the dataset.
