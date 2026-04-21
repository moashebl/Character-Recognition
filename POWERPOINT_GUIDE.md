# PowerPoint Writing Guide for ANN Character Recognition Project

Date: April 13, 2026

This guide gives you exactly what to write in your PowerPoint, slide by slide.
Use this as a script: copy the bullet points, add the visuals, and use the speaker notes while presenting.

---

## 1) Recommended Deck Length and Timing

- Target total slides: 10 to 12
- Target duration: 8 to 12 minutes
- Suggested timing split:
  - Introduction and problem: 1.5 minutes
  - Data and preprocessing: 2 minutes
  - Model and training method: 2 minutes
  - Results and analysis: 3 minutes
  - Demo and conclusion: 2 minutes

If your instructor gave a strict time limit, reduce details on theory slides and keep results/demo unchanged.

---

## 2) Slide-by-Slide Content (What to Write)

## Slide 1 - Title Slide

### Write this on the slide
- **Title:** Handwritten Character Recognition Using a NumPy MLP
- **Subtitle:** End-to-End ANN System with Training, Evaluation, and GUI Inference
- Your name
- Course name / instructor
- Date

### Speaker notes (what to say)
- This project builds a full pipeline for recognizing handwritten English letters A to Z.
- The system is implemented from scratch using NumPy and includes a GUI for real-time testing.

---

## Slide 2 - Problem Statement and Objective

### Write this on the slide
- **Problem:** Automatically classify handwritten letters A-Z.
- **Objective:** Build a complete ANN workflow from data loading to prediction.
- **Scope:**
  - Train a multi-class MLP model
  - Evaluate with objective metrics
  - Provide an interactive GUI for user input and prediction

### Speaker notes
- Focus on the word "complete": not just model code, but also data pipeline, analysis, and user-facing interface.
- Mention that the model is a baseline ANN and intentionally from scratch.

---

## Slide 3 - Project Pipeline Overview

### Write this on the slide
- Dataset (.npz) -> Preprocessing -> MLP Training -> Evaluation -> GUI Prediction
- Key files:
  - train_mlp.py
  - gui_app.py
  - src/ann/data.py
  - src/ann/mlp.py
  - src/ann/utils.py

### Speaker notes
- Briefly explain each stage in one sentence.
- Emphasize reproducibility and modular structure.

### Visual to include
- Add the architecture/training flow image from outputs folder.

---

## Slide 4 - Dataset Overview

### Write this on the slide
- **Dataset file:** character_fonts (with handwritten data).npz
- **Total samples:** 762,213 images
- **Image size:** 28 x 28 grayscale
- **Classes:** 26 (A-Z)
- **Split method:** Stratified train/test split with seed = 42

### Speaker notes
- Mention that class counts are not perfectly balanced.
- Explain why stratified split is important for fair evaluation.

---

## Slide 5 - Preprocessing Strategy

### Write this on the slide
- Normalize pixel values to [0, 1]
- Flatten each image to 784 features
- Remap labels to contiguous range 0 to 25
- GUI image preparation:
  - Convert to grayscale
  - Detect foreground and crop to ink region
  - Center in padded square
  - Resize to 28 x 28

### Speaker notes
- Explain that good preprocessing improves model performance as much as architecture changes.
- Highlight that GUI preprocessing aligns user drawings with training distribution.

---

## Slide 6 - Model Architecture and Learning

### Write this on the slide
- **Model:** Multi-Layer Perceptron (MLP)
- **Architecture:** 784 -> 256 -> 128 -> 26
- **Hidden activation:** ReLU
- **Output:** Softmax
- **Loss:** Cross-entropy
- **Optimization:** Mini-batch gradient descent with backpropagation

### Speaker notes
- Explain why MLP is suitable as a baseline: easy to implement and interpret.
- Mention that CNNs are usually stronger for image tasks, but this work focuses on ANN fundamentals.

---

## Slide 7 - Training Configuration

### Write this on the slide
- Learning rate around 0.01
- Batch size 256
- Epochs up to 20
- Early stopping support (patience based)
- Option to use max-samples for faster experimentation

### Speaker notes
- Explain that these settings balance speed and performance.
- Mention that early stopping helps avoid overfitting and unnecessary computation.

---

## Slide 8 - Quantitative Results

### Write this on the slide
- **Test samples:** 152,452
- **Test accuracy:** 0.7815
- **Test loss:** 0.8996
- **Saved pretrained model:** models/mlp_az.npz

### Speaker notes
- State clearly: these are measured results on a reproducible split, not estimated values.
- Compare expectations: this is good for a from-scratch MLP baseline.

### Visual to include
- Add training curve image: outputs/mlp_training_curve.png

---

## Slide 9 - Confusion Matrix and Error Analysis

### Write this on the slide
- Frequent confusion pairs:
  - D -> O
  - Y -> V
  - F -> P
  - I -> J
  - Q -> O
- Main reasons:
  - Similar letter shapes
  - Class imbalance
  - No data augmentation in baseline

### Speaker notes
- Explain that confusion analysis is more useful than only reporting accuracy.
- Show one or two examples where handwritten style can make letters nearly identical.

### Visual to include
- Add confusion matrix image: outputs/mlp_confusion_matrix.png

---

## Slide 10 - GUI Demonstration (Live or Screenshot)

### Write this on the slide
- GUI capabilities:
  - Train with adjustable hyperparameters
  - Load saved model
  - Draw letter on canvas and predict
  - Upload image and view top probabilities

### Speaker notes
- Do a short demo if possible:
  1. Open GUI
  2. Load pretrained model
  3. Draw a letter
  4. Show predicted class and confidence
- Keep this under 90 seconds.

---

## Slide 11 - Limitations and Future Work

### Write this on the slide
- **Current limitations:**
  - MLP is weaker than CNN for image recognition
  - Class imbalance affects minority classes
  - No augmentation (rotation/shift/thickness)
- **Future improvements:**
  - Move to CNN architecture
  - Add augmentation pipeline
  - Apply class weighting or focal loss
  - Add top-k metrics and calibration checks

### Speaker notes
- This slide shows critical thinking and engineering maturity.
- Make it clear you know how to improve the system beyond baseline.

---

## Slide 12 - Conclusion

### Write this on the slide
- Built a complete ANN-based character recognition pipeline from scratch.
- Achieved reproducible baseline performance on large handwritten dataset.
- Delivered both training/evaluation scripts and user-friendly GUI.
- Established clear path for next improvements (CNN + augmentation).

### Speaker notes
- End with impact: this project is not only theoretical, it is usable and extendable.

---

## 3) Visual Assets Checklist (Use in Slides)

Insert these files into your slides:

- outputs/mlp_training_curve.png
- outputs/mlp_confusion_matrix.png
- outputs/neural_network_architecture.png
- outputs/neural_network_training_flow.png
- outputs/logic_gate_and.png
- outputs/logic_gate_or.png
- outputs/logic_gate_xor.png

Optional sample input image:

- Images/Images/image.jpg

---

## 4) What Your Teacher/Panel Usually Looks For

Make sure your talk answers these five questions clearly:

1. What problem did you solve?
2. How did you design the model and pipeline?
3. How did you evaluate it fairly?
4. What do the errors tell you?
5. What would you improve next and why?

If these five points are clear, your presentation is strong even if the model is not state-of-the-art.

---

## 5) Common Presentation Mistakes to Avoid

- Too much math text on one slide.
- Showing only accuracy without confusion analysis.
- Explaining code line-by-line instead of design decisions.
- Long demo that can fail due to time or environment issues.
- No future work section.

---

## 6) 60-Second Backup Summary (If Time Is Cut)

Use this quick script:

"I built a full handwritten A-Z recognition system using a from-scratch NumPy MLP and a Tkinter GUI. The pipeline covers data loading, preprocessing, training, evaluation, and live inference. On a reproducible test split of 152,452 samples, the model achieved 78.15% accuracy with 0.8996 loss. Confusion matrix analysis shows errors mainly between visually similar letters like D/O and Y/V. The current model is a strong ANN baseline, and the next step is upgrading to CNNs with augmentation for better accuracy."

---

## 7) Final Preparation Checklist Before Presenting

- Verify all images load in your slides.
- Put large numbers in bold (samples, accuracy, loss).
- Rehearse timing once with a stopwatch.
- Test the GUI once before class.
- Keep one backup slide with extra details in case of questions.
