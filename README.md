# Neural Network Character Recognition

This project is a submission-focused, from-scratch NumPy ANN for handwritten character recognition.
It is simplified to the core requested workflow: train and run a GUI using the handwritten dataset.

## Submission Scope

- Main dataset: `character_fonts (with handwritten data).npz`
- Main model type: Multi-layer Perceptron (MLP)
- Main interface: `gui_app.py`

## Implemented Components

- Multi-layer perceptron (MLP) with backpropagation for multi-class character classification.
- Tkinter GUI to train/test the MLP with adjustable hyperparameters and live visual feedback.
- NPZ-based dataset loading and train/test split.

## Project Structure

- `src/ann/data.py`: dataset loading and train/test split.
- `src/ann/mlp.py`: from-scratch MLP classifier.
- `src/ann/utils.py`: encoding and evaluation helpers.
- `train_mlp.py`: train and evaluate multiclass MLP.
- `gui_app.py`: interactive training and prediction GUI with live plots.

## Install

```bash
c:/python313/python.exe -m pip install -r requirements.txt
```

## Train MLP (A-Z)

```bash
c:/python313/python.exe train_mlp.py --dataset "character_fonts (with handwritten data).npz" --epochs 15 --max-samples 120000 --val-size 0.1 --early-stopping-patience 5
```

The trainer now detects number of classes dynamically from the dataset labels.
If the dataset includes `class_names` in NPZ metadata, predictions and confusion matrix labels use those names.

Outputs:

- Model: `models/mlp_az.npz`
- Curves: `outputs/mlp_training_curve.png`
- Confusion matrix: `outputs/mlp_confusion_matrix.png`

## Run GUI

```bash
c:/python313/python.exe gui_app.py
```

In the GUI, you can:

- Pick a dataset (`.npz`).
- Set hidden layers, learning rate, epochs, batch size, max samples, validation split, and early stopping.
- Train the MLP with live loss/accuracy curves.
- Draw a character on canvas and predict it.
- Upload an image and view top class probabilities.

## Notes

- The full dataset is large; start with `--max-samples` to iterate faster.
- The MLP supports `relu` or `sigmoid` hidden activation.
- GUI training uses a background worker thread with safe main-thread UI updates.
