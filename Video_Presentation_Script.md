# Video Presentation Script: Neural Network-Based Character Recognition 

**Target Duration:** 5 - 10 minutes
**Visual Style:** Screen recording with voiceover (webcam in the corner optional but recommended).

---

## 1. Introduction & Project Definition (0:00 - 1:30)

**[Visual: Title Slide showing "Neural Network-Based Character Recognition Across Multiple Fonts", your name (Mohamed Alaa Shebl Mohamed), and Istanbul Atlas University logo.]**

**Speaker (You):** 
"Hello everyone, my name is Mohamed Alaa Shebl Mohamed. I am currently a student at Istanbul Atlas University, and today I'll be presenting my final project for the Artificial Neural Networks course. 

The goal of this project is to build a complete neural network system capable of recognizing English alphabet characters—from A to Z—using grayscale images. But here is the catch: instead of relying on high-level machine learning libraries like TensorFlow or PyTorch, I built the Multi-Layer Perceptron (or MLP) completely from scratch using only NumPy. 

This means the entire mathematical foundation of the network, including the forward pass, cross-entropy loss calculation, backpropagation, weight initialization, and mini-batch gradient descent, was coded manually. The main objective was to demonstrate a deep, fundamental understanding of how neural networks actually learn behind the scenes."

---

## 2. The Dataset Used (1:30 - 2:45)

**[Visual: Switch to VS Code showing the project structure, highlighting `data.py` and the `character_fonts (with handwritten data).npz` file. Show some sample image grids of alphabets if possible.]**

**Speaker (You):** 
"Let’s talk about the data driving this model. I used a modified version of the Alphabet Characters Fonts Dataset from Kaggle, extended to include handwritten character data. 

The dataset is massive—it contains over 762,000 images. Every image is a 28 by 28 pixel grayscale square, which gives us 784 input pixels for our neural network. The labels are integer-encoded from 0 to 25 to represent the 26 letters of the alphabet.

**[Visual: Display a quick slide showing Dataset Statistics (762k images, 26 classes) and a small bar chart or bullet point noting Class Imbalance.]**

One interesting challenge with this dataset is class imbalance. The most frequent character appears about four and a half times more often than the least frequent one. To prevent our model from becoming biased during training, I implemented a stratified train-test split to ensure that all classes are proportionally represented across our training and testing sets."

---

## 3. Results Obtained (2:45 - 4:15)

**[Visual: Show the `outputs/mlp_training_curve.png` and `outputs/mlp_confusion_matrix.png` on the screen side-by-side or one after the other.]**

**Speaker (You):** 
"So, how did the model perform? 

I designed the default architecture with an input layer of 784 neurons, two hidden layers of 256 and 128 neurons utilizing ReLU activation, and an output layer of 26 neurons using Softmax activation. 

Across our test set of over 150,000 unseen images, the from-scratch MLP achieved an accuracy of 78.15%. For a raw multi-layer perceptron reading flattened spatial pixels, this is a very solid baseline. 

If we look at the confusion matrix, the model’s mistakes actually make perfect visual sense. For example, it occasionally confuses 'D' with 'O', 'I' with 'J', and 'F' with 'P'. When you are dealing with low-resolution handwritten text, these letters naturally share very similar stroke structures.

I also created a bonus demonstration using this exact same MLP framework to solve logic gates—including the famous XOR problem—to prove that the network correctly utilizes hidden layers to solve non-linearly separable problems."

---

## 4. Live Demo & Interpretations (4:15 - 8:30)

**[Visual: Launch the GUI application by running `python gui_app.py` in the terminal. Show the interface cleanly on screen.]**

**Speaker (You):** 
"Now, let's look at the project in action. I built an interactive Tkinter graphical user interface to make this network easy to experiment with.

On the left side, we have our hyperparameters. I can dynamically adjust the number of hidden layers, the learning rate, the batch size, and toggle between activation functions like ReLU and Sigmoid. 

**[Visual: Demonstrate clicking around the GUI. Navigate to the "Training Metrics" tab.]**

If I click 'Train Model', the network begins mini-batch gradient descent in a background thread. You can see the loss and accuracy curves updating lived dynamically in the Training Metrics tab. It features early stopping, meaning if the validation loss stops improving, the network will automatically halt to prevent overfitting and restore the best weights.

**[Visual: Switch to the "Draw and Predict" tab. Use the mouse to draw a nice, clear letter, like an 'A' or 'B'. Click predict.]**

Let's test its predictive power right now. I'll draw the letter 'A' on this 320x320 canvas. When I click 'Predict', the GUI processes my drawing—cropping the ink, centering it, applying auto-contrast, and scaling it down to 28x28 pixels to match our training data. 

As you can see, the model correctly predicts 'A' with a high confidence score. In the Probability View tab, we can even see its second and third guesses.

**[Visual: Draw a messy letter that could be interpreted as two things, like a muddy 'O' or 'D', or an 'I' vs 'J'.]**

Let's make a change and draw something ambiguous. Here it is a bit messy, somewhere between a 'U' and a 'V' (or 'D' and 'O'). Notice how the confidence drops, and the probability bar charts show the model deciding between the two classes. This interprets exactly what we saw in the confusion matrix earlier—the neural network is distinguishing edges and curves, and ambiguous pixels split its confidence."

---

## 5. Conclusion & Future Work (8:30 - 9:30)

**[Visual: Slide summarizing "Conclusion & Future Work" (CNNs, Data Augmentation).]**

**Speaker (You):** 
"In conclusion, this project successfully proves that we can build a robust, capable neural network completely from mathematical first principles in Python.

While an accuracy of 78% is great for an MLP, flattened arrays lose spatial awareness. If I were to improve this system in the future, the natural next step would be to introduce Convolutional Neural Networks (CNNs) to extract 2D spatial features better. I would also introduce data augmentation—like random rotations and shifting—to make the model even more robust to messy handwriting.

Thank you for watching my presentation, and I hope you enjoyed this look under the hood of Artificial Neural Networks!"

---

### Tips for Recording:
* **Practice the Demo:** Run through the GUI demo a few times before recording to ensure you know exactly how the model will react to your drawings.
* **Smooth Transitions:** Pause slightly between sections so that if you need to edit the video later, you have clean breaks to cut the audio cleanly.
* **Cursor Movement:** Hide your taskbar and move your mouse deliberately when highlighting things in the GUI to make it easy for the viewer to follow.