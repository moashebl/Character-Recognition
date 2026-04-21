# Neural Network-Based Character Recognition Across Multiple Fonts
## Project Report

**Student:** Mohamed Alaa Shebl Mohamed  
**Student ID:** 230504583  
**Course:** Artificial Neural Networks — Istanbul Atlas University  
**Submission Date:** April 2025  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Project Objective](#2-project-objective)
3. [Dataset Description](#3-dataset-description)
4. [Project Structure and File Organization](#4-project-structure-and-file-organization)
5. [Methodology and Neural Network Design](#5-methodology-and-neural-network-design)
   - 5.1 Multi-Layer Perceptron (MLP) Architecture
   - 5.2 Activation Functions
   - 5.3 Loss Function (Cross-Entropy)
   - 5.4 Backpropagation Algorithm
   - 5.5 Weight Initialization
   - 5.6 Mini-Batch Gradient Descent
   - 5.7 Early Stopping
6. [Data Preprocessing Pipeline](#6-data-preprocessing-pipeline)
7. [Code Walkthrough](#7-code-walkthrough)
   - 7.1 src/ann/utils.py — Utility Functions
   - 7.2 src/ann/data.py — Data Loading Module
   - 7.3 src/ann/mlp.py — MLP Classifier (Core Neural Network)
   - 7.4 train_mlp.py — Command-Line Training Script
   - 7.5 gui_app.py — GUI Application
   - 7.6 generate_report_assets.py — Report Figure Generator
8. [Training Configuration and Hyperparameters](#8-training-configuration-and-hyperparameters)
9. [Results and Evaluation](#9-results-and-evaluation)
   - 9.1 Quantitative Results
   - 9.2 Training Curves
   - 9.3 Confusion Matrix Analysis
10. [Logic Gate Demonstrations (Bonus)](#10-logic-gate-demonstrations-bonus)
11. [GUI Application](#11-gui-application)
12. [Limitations and Future Work](#12-limitations-and-future-work)
13. [Conclusion](#13-conclusion)
14. [References](#14-references)

---

## 1. Introduction

Character recognition is one of the fundamental problems in pattern recognition and computer vision. The ability to automatically identify handwritten or printed characters has wide applications in postal mail sorting, bank check processing, document digitization, and assistive technologies.

This project tackles the problem of recognizing English alphabet characters (A–Z) using Artificial Neural Networks (ANNs), specifically a Multi-Layer Perceptron (MLP). The entire neural network — including forward propagation, loss computation, backpropagation, and weight updates — is implemented from scratch using NumPy, without relying on high-level ML libraries such as TensorFlow, PyTorch, or Scikit-Learn. This from-scratch approach was chosen to demonstrate a deep understanding of the mathematical foundations behind neural network learning.

In addition to the model itself, the project includes a graphical user interface (GUI) built with Tkinter that allows users to train models interactively, adjust hyperparameters in real time, draw characters on a canvas for prediction, and upload images for classification.

---

## 2. Project Objective

The main objective of this project is to build a complete neural network system that can:

1. **Recognize characters A–Z** from grayscale images using a from-scratch MLP implementation.
2. **Train and evaluate** the model on a large dataset of handwritten character images.
3. **Provide an interactive GUI** for training with live feedback, drawing-based prediction, and image upload prediction.
4. **Demonstrate core ANN concepts** such as activation functions, backpropagation, cross-entropy loss, softmax classification, and decision boundaries.
5. **Implement bonus demonstrations** of logic gates (AND, OR, XOR) using the same MLP framework to illustrate linear separability and the need for hidden layers.

---

## 3. Dataset Description

### 3.1 Source

The dataset used is based on the **Alphabet Characters Fonts Dataset** from Kaggle, enhanced with handwritten character data. It is stored in NumPy's compressed `.npz` format.

**File:** `character_fonts (with handwritten data).npz`

### 3.2 Dataset Statistics

| Property | Value |
|---|---|
| Total images | 762,213 |
| Image dimensions | 28 × 28 pixels |
| Color mode | Grayscale (uint8, 0–255) |
| Number of classes | 26 (A–Z) |
| Label format | Integer (0–25) |
| Minimum class count | 16,109 |
| Maximum class count | 72,816 |
| Mean class count | 29,315.88 |

### 3.3 NPZ Structure

The `.npz` file contains the following arrays:
- **`images`**: shape `(762213, 28, 28)` — grayscale pixel values as uint8
- **`labels`**: shape `(762213,)` — integer class labels

### 3.4 Class Imbalance

The dataset exhibits some class imbalance, with the most frequent class having approximately 4.5× more samples than the least frequent class. This imbalance can cause the model to be biased toward predicting more common classes. The stratified train/test split preserves this distribution in both sets.

---

## 4. Project Structure and File Organization

The project follows a clean, modular structure:

```
Project/
├── src/
│   └── ann/
│       ├── __init__.py          # Package exports
│       ├── data.py              # Dataset loading, splitting, preprocessing
│       ├── mlp.py               # MLP classifier (from-scratch neural network)
│       └── utils.py             # One-hot encoding, metrics, confusion matrix
├── models/
│   ├── mlp_az.npz               # Pretrained model (CLI training output)
│   └── gui_mlp_model.npz        # Pretrained model (GUI training output)
├── outputs/
│   ├── mlp_training_curve.png    # Loss and accuracy plots
│   ├── mlp_confusion_matrix.png  # Confusion matrix heatmap
│   ├── logic_gate_and.png        # AND gate decision boundary
│   ├── logic_gate_or.png         # OR gate decision boundary
│   ├── logic_gate_xor.png        # XOR gate decision boundary
│   ├── neural_network_architecture.png  # MLP architecture diagram
│   └── neural_network_training_flow.png # Training loop diagram
├── Images/
│   └── Images/                   # Sample character images organized by class (A-Z folders)
├── train_mlp.py                  # Command-line training entry point
├── gui_app.py                    # Interactive Tkinter GUI application
├── generate_report_assets.py     # Script to generate report figures
├── requirements.txt              # Python dependencies
├── README.md                     # Project README
└── character_fonts (with handwritten data).npz  # Main dataset
```

### 4.1 Module Descriptions

| File | Purpose | Lines of Code |
|---|---|---|
| `src/ann/utils.py` | Helper functions for encoding and evaluation | ~48 |
| `src/ann/data.py` | Data loading from NPZ and image folders, train/test splitting | ~157 |
| `src/ann/mlp.py` | Complete MLP implementation with forward, backward, fit, predict | ~284 |
| `train_mlp.py` | CLI-based model training and evaluation script | ~143 |
| `gui_app.py` | Interactive GUI with drawing canvas and live training plots | ~836 |
| `generate_report_assets.py` | Generates logic gate and architecture diagrams | ~232 |

---

## 5. Methodology and Neural Network Design

### 5.1 Multi-Layer Perceptron (MLP) Architecture

The MLP is a fully-connected feed-forward neural network. The default architecture used for character recognition is:

```
Input Layer (784 neurons) → Hidden Layer 1 (256 neurons) → Hidden Layer 2 (128 neurons) → Output Layer (26 neurons)
```

- **Input layer:** 784 neurons, corresponding to the flattened 28×28 pixel image
- **Hidden layer 1:** 256 neurons with ReLU activation
- **Hidden layer 2:** 128 neurons with ReLU activation
- **Output layer:** 26 neurons with Softmax activation (one per class A–Z)

**Total trainable parameters:** 237,210 (weights + biases across all layers)

The parameter count breakdown:
- Layer 1: 784 × 256 + 256 = 200,960
- Layer 2: 256 × 128 + 128 = 32,896
- Layer 3: 128 × 26 + 26 = 3,354
- **Total: 237,210**

### 5.2 Activation Functions

The project implements two activation functions:

**ReLU (Rectified Linear Unit):**
```
f(x) = max(0, x)
f'(x) = 1 if x > 0, else 0
```

ReLU is used as the default because it avoids the vanishing gradient problem that can slow down training with sigmoid in deep networks.

**Sigmoid:**
```
f(x) = 1 / (1 + e^(-x))
f'(x) = f(x) × (1 - f(x))
```

Sigmoid is available as an alternative and is used for the logic gate demonstrations.

**Softmax (Output Layer):**
```
softmax(z_i) = e^(z_i) / Σ_j e^(z_j)
```

The implementation uses the numerically stable version by subtracting the maximum value before exponentiation to prevent overflow.

### 5.3 Loss Function (Cross-Entropy)

The model uses categorical cross-entropy loss:

```
L = -(1/m) × Σ_n Σ_k y_nk × log(ŷ_nk)
```

Where:
- `m` is the number of samples in the batch
- `y_nk` is the one-hot encoded true label
- `ŷ_nk` is the predicted probability from softmax
- Clipping is applied to prevent log(0) errors

### 5.4 Backpropagation Algorithm

Backpropagation computes the gradient of the loss with respect to each weight and bias in the network, layer by layer from output to input:

1. **Output layer error:** `δ_output = ŷ - y` (simplified gradient for softmax + cross-entropy)
2. **Hidden layer error:** `δ_l = (δ_(l+1) × W_(l+1)^T) ⊙ f'(z_l)` where `⊙` is element-wise multiplication
3. **Weight gradients:** `∂L/∂W_l = (1/m) × a_(l-1)^T × δ_l`
4. **Bias gradients:** `∂L/∂b_l = (1/m) × Σ δ_l`
5. **Weight update:** `W_l = W_l - η × ∂L/∂W_l`

All of this is implemented manually in the `_backward()` method using NumPy matrix operations.

### 5.5 Weight Initialization

Proper weight initialization is important to ensure stable training:

- **For ReLU:** He initialization — `W ~ N(0, sqrt(2/fan_in))` — this accounts for the fact that ReLU zeros out half the activations.
- **For Sigmoid:** Xavier-like initialization — `W ~ N(0, sqrt(1/fan_in))` — keeps the variance of activations consistent across layers.
- **Biases** are initialized to zero.

### 5.6 Mini-Batch Gradient Descent

Instead of computing gradients on the entire dataset at once (which is memory-intensive) or on single samples (which is noisy), the training uses mini-batch gradient descent:

- Default batch size: 256
- Data is shuffled at the beginning of each epoch
- Gradients are computed and weights are updated for each mini-batch
- This provides a good balance between computational efficiency and gradient stability

### 5.7 Early Stopping

To prevent overfitting, the training loop supports early stopping:

- After each epoch, validation loss is evaluated
- If validation loss does not improve for a specified number of epochs (patience), training stops
- The model weights are reverted to the best-performing epoch
- Default patience: 5 epochs

---

## 6. Data Preprocessing Pipeline

### 6.1 Training Data Preprocessing

1. **Load from NPZ:** Images and labels are loaded from the `.npz` file
2. **Resize:** If images are not 28×28, they are resized using bilinear interpolation via PIL
3. **Normalize:** Pixel values are converted from uint8 (0–255) to float32 (0.0–1.0)
4. **Flatten:** Each 28×28 image is reshaped into a 784-element vector
5. **Label remapping:** Labels are mapped to contiguous integers 0–25 for consistent one-hot encoding

### 6.2 GUI Image Preprocessing

The GUI applies additional preprocessing for drawings and uploaded images:

1. **Grayscale conversion**
2. **Foreground detection:** Analyzes pixel intensity distribution to determine if the foreground is dark or light
3. **Cropping:** Identifies the bounding box of "ink pixels" and crops tightly around them
4. **Centering:** Centers the cropped character in a padded square with consistent margins
5. **Auto-contrast:** Enhances contrast using PIL's autocontrast
6. **Resize:** Scales to 28×28 using bilinear resampling

### 6.3 Polarity Handling

Since different datasets may have dark characters on light backgrounds or light characters on dark backgrounds, the prediction logic:

1. Runs prediction on both the original and inverted versions of the input
2. Compares confidence scores
3. Selects the version with higher confidence
4. Uses dataset polarity estimation as a hint when available

---

## 7. Code Walkthrough

### 7.1 src/ann/utils.py — Utility Functions

This file provides helper functions used throughout the project:

- **`one_hot_encode(labels, num_classes)`**: Converts integer labels into a one-hot encoded matrix. Each row has a single 1.0 at the position of the correct class. This is required for the cross-entropy loss calculation.

- **`labels_to_vowel_consonant(labels)`**: Converts A–Z labels into binary vowel/consonant labels (mapping A, E, I, O, U to 1 and the rest to 0). This was intended for the Perceptron binary classification experiment.

- **`accuracy_score(y_true, y_pred)`**: Calculates the proportion of correct predictions.

- **`classification_report(y_true, y_pred, num_classes)`**: Generates a per-class precision, recall, and F1-score report. All metrics are computed manually without external ML libraries. Precision measures how many of the model's positive predictions for a class were correct, recall measures how many of the actual samples in that class were found, and F1 is the harmonic mean of both.

- **`confusion_matrix(y_true, y_pred, num_classes)`**: Builds a matrix where entry (i, j) counts how many samples of true class i were predicted as class j. This is used for error analysis.

### 7.2 src/ann/data.py — Data Loading Module

This module handles all data input/output operations:

- **`_resize_images(images, target_size)`**: Internal helper that resizes a batch of images to the specified dimensions using PIL. Only runs if images need resizing, avoiding unnecessary computation.

- **`load_from_npz(npz_path, target_size, flatten, normalize)`**: Loads a dataset from an NPZ file. It reads the `images` and `labels` arrays, applies resizing, normalizes pixel values to [0, 1], and optionally flattens images into 1D vectors. This is the primary data loading function used by both the training script and the GUI.

- **`get_npz_class_names(npz_path)`**: Checks if the NPZ file contains a `class_names` array and returns it if present. This allows datasets to include human-readable labels.

- **`make_labels_contiguous(labels, class_names)`**: Ensures that labels form a contiguous range from 0 to K-1. This is necessary because some datasets may have gaps in their label indices. If 26 contiguous labels 0–25 are found, they are automatically named A–Z.

- **`load_from_image_folders(root_dir, ...)`**: An alternative data loader that reads images organized in per-class folders (e.g., A/, B/, C/, ...). It handles edge cases like nested folder structures and supports limiting samples per class.

- **`train_test_split(x, y, test_size, seed, stratify)`**: Splits data into training and testing sets. When stratify is True, each class is split separately to maintain class proportions in both sets. Uses a seeded random number generator for reproducibility.

### 7.3 src/ann/mlp.py — MLP Classifier (Core Neural Network)

This is the core of the project — the complete from-scratch MLP implementation:

**MLPHistory (dataclass):**
Stores training metrics per epoch: losses, accuracies, validation losses, validation accuracies, and the best epoch number.

**MLPClassifier class:**

- **`__init__(layer_sizes, learning_rate, epochs, batch_size, hidden_activation, seed, class_names)`**: Initializes the network by creating weight matrices and bias vectors for each layer connection. Weight initialization uses He scaling for ReLU and Xavier scaling for sigmoid.

- **`_hidden_activation(x)`**: Applies the configured activation function (ReLU or sigmoid) element-wise to the input array.

- **`_hidden_activation_derivative(x)`**: Computes the derivative of the activation function, which is needed during backpropagation to calculate how much each neuron contributed to the error.

- **`_softmax(x)`** (static method): Computes softmax probabilities in a numerically stable way by subtracting the row-wise maximum before exponentiation.

- **`_cross_entropy(y_true_one_hot, y_pred_probs)`** (static method): Computes the average cross-entropy loss, with clipping to prevent log(0).

- **`forward(x)`**: Executes the complete forward pass through all layers. For each hidden layer, it computes `z = a·W + b` and applies the activation function. The final layer applies softmax instead. Returns both the activations (outputs of each layer) and pre-activations (raw linear outputs before activation), both of which are needed for backpropagation.

- **`_backward(activations, pre_activations, y_true_one_hot)`**: Implements full backpropagation. Starts with the output error (predicted - true), then propagates backward through each layer computing weight and bias gradients using the chain rule and the stored pre-activations.

- **`fit(x, y, x_val, y_val, early_stopping_patience, verbose, epoch_callback)`**: The main training loop. For each epoch, it shuffles the data, processes it in mini-batches (forward pass → backward pass → weight update), computes epoch-level loss and accuracy, optionally evaluates on validation data, and checks for early stopping. The `epoch_callback` parameter allows the GUI to receive progress updates safely from the training thread.

- **`predict_proba(x)`**: Returns softmax probabilities for each input sample.

- **`predict(x)`**: Returns the class index with the highest probability for each sample.

- **`evaluate(x, y)`**: Computes both the cross-entropy loss and accuracy on a given dataset.

- **`save(path)`**: Saves the model to an NPZ file, including layer sizes, hyperparameters, all weight/bias arrays, and class names.

- **`load(path)`** (class method): Reconstructs an MLPClassifier from a saved NPZ file, restoring all weights and configuration.

### 7.4 train_mlp.py — Command-Line Training Script

This script provides a command-line interface for training:

- **`parse_args()`**: Defines all command-line arguments including dataset path, hidden layer sizes, learning rate, epochs, batch size, activation function, max samples, validation size, early stopping patience, and output paths.

- **`main()`**: Orchestrates the full training pipeline:
  1. Loads the dataset from NPZ
  2. Extracts class names and remaps labels to contiguous indices
  3. Optionally subsamples the dataset for faster experimentation
  4. Splits into train/test and optionally train/val sets
  5. Creates the MLP model
  6. Trains with optional early stopping and verbose output
  7. Evaluates on the test set and prints a classification report
  8. Saves the model weights
  9. Generates and saves the training curve plot (loss + accuracy)
  10. Generates and saves the confusion matrix visualization

### 7.5 gui_app.py — GUI Application

The GUI application provides a complete interactive interface:

**ANNGui class (inherits from tk.Tk):**

- **Layout:** The window uses a horizontal PanedWindow with a left panel (configuration/controls) and a right panel (results/visualization). The right panel contains a Notebook with three tabs: "Draw and Predict", "Training Metrics", and "Probability View".

- **Left panel components:**
  - Dataset selection with file browser
  - Hyperparameter fields: hidden layers, learning rate, epochs, batch size, max samples, validation size, early stopping patience, and activation function selector
  - Action buttons: Train Model, Load Model, Predict Image
  - Training progress bar
  - Scrollable status log

- **Right panel tabs:**
  - **Draw and Predict tab:** A 320×320 canvas where users can draw characters with adjustable brush size, plus a prediction details section showing the predicted label, confidence, processed 28×28 preview, and a top-predictions table.
  - **Training Metrics tab:** Live-updating Matplotlib plots showing loss and accuracy curves for both training and validation sets.
  - **Probability View tab:** A horizontal bar chart showing the top class probabilities after each prediction.

- **Thread safety:** Training runs in a background thread to keep the GUI responsive. The worker thread sends events through a `Queue`, and the main thread polls this queue every 80ms using Tkinter's `after()` mechanism. This avoids thread-safety issues with Tkinter's single-threaded event loop.

- **Key workflows:**
  - **Training:** User sets hyperparameters → clicks "Train Model" → background thread loads data, trains the MLP with epoch callbacks → GUI updates progress bar and live plots → model is saved to `models/gui_mlp_model.npz`
  - **Drawing prediction:** User draws on canvas → clicks "Predict Drawing" → image is preprocessed → prediction runs on both original and inverted versions → best result is displayed
  - **Image prediction:** User uploads an image file → image is preprocessed → prediction runs → results displayed

### 7.6 generate_report_assets.py — Report Figure Generator

This script generates visual assets for the report and presentation:

- **`generate_logic_gate_images()`**: Trains small 2-input MLP models on AND, OR, and XOR truth tables. For each gate, it creates a visualization showing the decision surface (probability contour), the data points with true labels, and a truth table. The XOR example is particularly important as it demonstrates why hidden layers are necessary.

- **`generate_neural_network_diagrams()`**: Creates two conceptual diagrams:
  - An architecture diagram showing the layers of the MLP (Input  → Hidden 1 → Hidden 2 → Output) with node counts
  - A training flow diagram showing the steps of one training iteration (Input → Preprocess → Forward Pass → Softmax → Loss → Backprop → Update)

---

## 8. Training Configuration and Hyperparameters

The default training configuration for the shipped model:

| Parameter | Value |
|---|---|
| Architecture | 784 → 256 → 128 → 26 |
| Hidden activation | ReLU |
| Output activation | Softmax |
| Loss function | Cross-entropy |
| Learning rate | 0.01 |
| Batch size | 256 |
| Max epochs | 20 |
| Early stopping patience | 5 |
| Validation split | 10% of training data |
| Test split | 20% of full data |
| Random seed | 42 |
| Max samples (CLI) | 120,000 |

---

## 9. Results and Evaluation

### 9.1 Quantitative Results

The trained model was evaluated on a reproducible stratified test split:

| Metric | Value |
|---|---|
| Dataset size | 762,213 images |
| Test set size | 152,452 images |
| Test accuracy | 0.7815 (78.15%) |
| Test cross-entropy loss | 0.8996 |
| Total parameters | 237,210 |

The accuracy of 78.15% is a reasonable result for a from-scratch MLP baseline on raw pixel inputs. State-of-the-art CNN models typically achieve 90%+ on similar tasks, but this project focuses on understanding ANN fundamentals rather than maximizing performance.

### 9.2 Training Curves

The training curve plot (saved to `outputs/mlp_training_curve.png`) shows:

- **Left subplot:** Training and validation loss decreasing over epochs, indicating the model is learning
- **Right subplot:** Training and validation accuracy increasing, with validation accuracy closely tracking training accuracy, suggesting reasonable generalization

### 9.3 Confusion Matrix Analysis

The confusion matrix (saved to `outputs/mlp_confusion_matrix.png`) reveals the most common misclassifications:

| True Class | Predicted As | Count | Reason |
|---|---|---|---|
| D | O | 474 | Similar round shapes |
| Y | V | 336 | Similar angular appearance |
| F | P | 317 | Similar top structure |
| I | J | 315 | Similar vertical strokes |
| Q | O | 312 | Q looks like O without tail |
| H | N | 298 | Similar vertical and horizontal strokes |
| E | C | 280 | Similar curved structure |
| O | D | 266 | Reverse of D→O confusion |
| Y | T | 254 | Similar top part |
| S | J | 245 | Curving patterns |
| U | W | 242 | Similar bottom curves |

These confusions make visual sense — the misclassified letters genuinely look similar when handwritten, especially in noisy or low-resolution character images.

---

## 10. Logic Gate Demonstrations (Bonus)

As a bonus component, the project demonstrates that the same MLP implementation can solve classic logic gate problems:

### AND Gate
- **Truth table:** (0,0)→0, (0,1)→0, (1,0)→0, (1,1)→1
- **Linearly separable:** Yes — a single hyperplane can separate the classes
- **Result:** The trained MLP correctly classifies all 4 inputs

### OR Gate
- **Truth table:** (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→1
- **Linearly separable:** Yes
- **Result:** Correctly classified

### XOR Gate
- **Truth table:** (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
- **Linearly separable:** No — this is the key demonstration
- **Result:** The MLP with a hidden layer successfully learns XOR, showing why multi-layer networks are needed for non-linearly separable problems

The visualization for each gate shows the decision surface (probability contour), the data points, and the truth table.

---

## 11. GUI Application

The GUI (`gui_app.py`) provides a user-friendly interface that includes:

### 11.1 Features

1. **Dataset Selection:** Browse and select any `.npz` dataset file
2. **Hyperparameter Configuration:** Adjust hidden layers, learning rate, epochs, batch size, max samples, validation size, early stopping, and activation function
3. **Training with Live Feedback:** Train the model and watch loss/accuracy curves update in real time
4. **Drawing Canvas:** A 320×320 pixel canvas with adjustable brush size for drawing characters
5. **Image Upload:** Load external image files for prediction
6. **Prediction Display:** Shows predicted class, confidence percentage, processed 28×28 preview, and top-k predictions
7. **Probability Visualization:** Horizontal bar chart showing class probabilities

### 11.2 Design Decisions

- **Thread safety:** Training runs in a background thread with a queue-based event system to prevent the GUI from freezing during long training sessions
- **Polarity detection:** Automatically detects whether the dataset uses dark or light backgrounds and adjusts predictions accordingly
- **Dual prediction:** Tries both normal and inverted input images and picks the one with higher confidence, improving robustness
- **Status log:** A scrollable text area shows all training progress and prediction results, with automatic trimming to prevent memory growth

---

## 12. Limitations and Future Work

### 12.1 Current Limitations

1. **MLP vs. CNN:** An MLP operating on flattened pixels cannot capture spatial features. Convolutional Neural Networks (CNNs) are significantly better for image classification tasks because they preserve spatial structure.
2. **Class imbalance:** The dataset has uneven class counts, which can bias the model toward predicting more common classes.
3. **No data augmentation:** The training pipeline does not apply transformations like rotation, shifting, scaling, or noise addition, which could improve generalization.
4. **Fixed learning rate:** The learning rate does not decay over time, which can prevent fine-tuning in later epochs.

### 12.2 Future Improvements

1. **Architecture upgrade:** Implement a CNN with convolutional and pooling layers for better spatial feature extraction
2. **Data augmentation:** Add random rotations, translations, thickness variations, and noise to the training pipeline
3. **Class balancing:** Apply class weighting in the loss function or use oversampling/undersampling techniques
4. **Learning rate scheduling:** Implement learning rate decay or use adaptive optimizers like Adam
5. **Regularization:** Add dropout layers or L2 weight regularization to reduce overfitting
6. **Extended character set:** Support lowercase letters, digits, and special characters

---

## 13. Conclusion

This project successfully demonstrates the construction of a complete neural network-based character recognition system from scratch. The key accomplishments are:

1. **From-scratch implementation:** The entire MLP — including forward propagation, backpropagation, softmax, cross-entropy, mini-batch gradient descent, and early stopping — is implemented using only NumPy.
2. **Functional model:** The trained model achieves 78.15% accuracy on a test set of 152,452 handwritten character images across 26 classes.
3. **Interactive GUI:** A full Tkinter-based application allows users to train models, adjust hyperparameters, draw characters, and predict from uploaded images.
4. **Educational demonstrations:** Logic gate examples visually illustrate concepts of linear separability and the value of hidden layers.
5. **Reproducible results:** Fixed random seeds and stratified splits ensure that all reported metrics can be independently verified.

The project meets the course requirements for implementing a neural network from scratch with clear documentation, evaluation, and a user-facing interface.

---

## 14. References

1. Fausett, L. (1994). *Fundamentals of Neural Networks: Architectures, Algorithms, and Applications*. Prentice Hall.
2. Kaggle — Alphabet Characters Fonts Dataset: kaggle.com/datasets/thomasqazwsxedc/alphabet-characters-fonts-dataset
3. NumPy Documentation: numpy.org/doc/stable/
4. Tkinter Documentation: docs.python.org/3/library/tkinter.html
5. Matplotlib Documentation: matplotlib.org/stable/
6. He, K., et al. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." *ICCV 2015*.
7. Glorot, X. & Bengio, Y. (2010). "Understanding the difficulty of training deep feedforward neural networks." *AISTATS 2010*.

---

*This report was prepared by Mohamed Alaa Shebl Mohamed for the Artificial Neural Networks course at Istanbul Atlas University.*
