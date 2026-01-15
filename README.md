# 🔢 Handwritten Digit Recognizer with CNN

> A deep learning project achieving **99.5% accuracy** on the MNIST digit recognition task using TensorFlow and Convolutional Neural Networks.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://python.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/code/rishabhkannaujiya/99-5-accuracy-cnn-tensorflow)

---

## 🎯 Overview

This project implements a state-of-the-art Convolutional Neural Network (CNN) to recognize handwritten digits from the famous MNIST dataset. The model achieves exceptional accuracy through advanced techniques including data augmentation, batch normalization, and dynamic learning rate adjustment.

### What makes this special?

- **High Performance**: 99.5% validation accuracy
- **Production Ready**: Complete training pipeline with checkpointing and callbacks
- **Interactive Visualizations**: Feature maps, confusion matrices, and error analysis
- **Well Documented**: Clean, modular code with detailed comments

---

## 🏗️ Model Architecture

```
Input (28×28×1)
    ↓
┌─────────────────────────────────────┐
│  Block A: Feature Detection         │
│  • Conv2D (32 filters, 5×5)         │
│  • BatchNormalization               │
│  • Conv2D (32 filters, 5×5)         │
│  • BatchNormalization               │
│  • MaxPooling2D (2×2)               │
│  • Dropout (0.25)                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Block B: Complex Patterns          │
│  • Conv2D (64 filters, 3×3)         │
│  • BatchNormalization               │
│  • Conv2D (64 filters, 3×3)         │
│  • BatchNormalization               │
│  • MaxPooling2D (2×2)               │
│  • Dropout (0.25)                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Block C: Classification            │
│  • Flatten                          │
│  • Dense (256 units)                │
│  • BatchNormalization               │
│  • Dropout (0.5)                    │
│  • Dense (10 units, softmax)        │
└─────────────────────────────────────┘
    ↓
Output (10 classes)
```

**Total Parameters**: ~1.2M trainable parameters

---

## 🔑 Key Techniques

### 1. Data Augmentation

```python
ImageDataGenerator(
    rotation_range=10,      # Rotate images ±10°
    zoom_range=0.1,         # Zoom in/out 10%
    width_shift_range=0.1,  # Shift horizontally
    height_shift_range=0.1  # Shift vertically
)
```

**Why?** Prevents overfitting by creating variations of training images.

### 2. Batch Normalization

Normalizes layer inputs, leading to:
- Faster training
- Better gradient flow
- Improved accuracy

### 3. Dynamic Learning Rate

```python
ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=3,
    factor=0.5
)
```

**Why?** Automatically reduces learning rate when validation accuracy plateaus.

### 4. Dropout Regularization

- 25% dropout after convolutional blocks
- 50% dropout before final layer

**Why?** Prevents overfitting by randomly deactivating neurons during training.

---

## 📊 Visualizations

### Training History

The notebook generates comprehensive training visualizations:

- **Loss Curves**: Track training vs validation loss
- **Accuracy Curves**: Monitor model improvement
- **Confusion Matrix**: Detailed class-wise performance

### Feature Map Visualization

See what the CNN "sees" at each layer:

- **Early Layers**: Detect edges and simple patterns
- **Middle Layers**: Recognize curves and loops
- **Deep Layers**: Identify complete digit shapes

### Error Analysis

Visual inspection of misclassified digits helps understand:
- Common confusion pairs (e.g., 4 vs 9)
- Ambiguous handwriting cases
- Areas for model improvement

---

<div align="center">

### ⭐ If you found this project helpful, please consider giving it a star!

**Made with ❤️ and TensorFlow**

</div>
