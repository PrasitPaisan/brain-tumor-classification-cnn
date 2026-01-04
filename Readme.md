# 🧠 Brain Tumor Classification using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-InProgress-yellow)]()

> **"Early detection saves lives."**
> A Deep Learning project designed to classify MRI scans into different types of brain cancer using a custom VGG-inspired Convolutional Neural Network (CNN).

---

## 📸 Demo Preview

![Project Demo](https://github.com/PrasitPaisan/brain-cancer-classification-cnn/blob/69c848aa68f4a70d90fe24be573db3f0c6f1b2fc/img_info/banner.png)

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Performance](#-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🧐 Overview

This project aims to develop an AI model to assist medical professionals in diagnosing and classifying brain cancer from MRI scans. By automating the initial feature extraction and classification process, this tool seeks to reduce diagnostic time and improve screening accuracy.

**Key Features:**
- ✅ **Multi-class Classification:** Capable of classifying 3 distinct tumor types (Glioma, Meningioma, Pituitary).
- ✅ **Custom Architecture:** Utilizes a **VGG-inspired CNN** specifically tuned to extract hierarchical features from grayscale MRI images.
- ✅ **Robustness:** Integrated Data Augmentation pipeline to mitigate overfitting and improve model generalization on unseen data.

---

## 📂 Dataset

The model was trained on the [Multi Cancer Dataset (Kaggle)](https://www.kaggle.com/datasets/obulisainaren/multi-cancer), consisting of approximately **15,000 MRI images** divided into three classes:

1. **Glioma Tumor**
   - A common type of tumor originating in the glial cells of the brain or spine.
2. **Meningioma Tumor**
   - A tumor that forms on membranes that cover the brain and spinal cord just inside the skull.
3. **Pituitary Tumor**
   - A tumor that forms in the pituitary gland at the base of the brain.

![Dataset Samples](https://github.com/PrasitPaisan/brain-cancer-classification-cnn/blob/69c848aa68f4a70d90fe24be573db3f0c6f1b2fc/img_info/brain_cancer.png)

---

## 🏗 Model Architecture

We utilized a **Custom CNN architecture inspired by VGG**, designed to effectively extract hierarchical features from MRI scans while maintaining computational efficiency. The model employs a block-based structure characterized by repeated convolutional layers followed by pooling.

### Implementation Details
The architecture consists of 3 main Convolutional Blocks followed by a Fully Connected classifier.

```python
model = keras.Sequential([
    # Input & Preprocessing
    layers.Input(shape=Image_Size + (1,)),
    layers.Rescaling(1./255),
    Augmentation,

    # Block 1 (VGG-Style)
    layers.Conv2D(32, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Dropout(0.25),

    # Block 2 (VGG-Style)
    layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Dropout(0.25),

    # Block 3 (VGG-Style)
    layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"),
    layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Dropout(0.25),

    # Classification Head
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

```

### 📊 Performance

The model was evaluated on an independent test set of 2,253 MRI images. The results demonstrate high accuracy and robust generalization, with minimal overfitting as shown in the loss curves.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **97.7%** |
| Precision (Avg) | 98.0% |
| Recall (Avg) | 98.0% |
| F1-Score | 0.98 |

### Training History
The training and validation accuracy curves rise steadily, converging around 97-98%. The loss curves decrease in tandem, indicating that the `Dropout` and `Data Augmentation` strategies effectively prevented overfitting.

![Training History](https://github.com/PrasitPaisan/brain-cancer-classification-cnn/blob/4259b2241af5d1d38f87c5767e38a7e83d578d7a/img_info/loss_and_accuracy_graph.png)

### Confusion Matrix
The confusion matrix highlights the model's ability to distinguish between cancer types with high precision.
- **Glioma:** 98% accuracy
- **Meningioma:** 96% accuracy
- **Pituitary (brain_tumor):** 99% accuracy

![Confusion Matrix](https://github.com/PrasitPaisan/brain-cancer-classification-cnn/blob/69c848aa68f4a70d90fe24be573db3f0c6f1b2fc/img_info/confusion-metrix.png)
