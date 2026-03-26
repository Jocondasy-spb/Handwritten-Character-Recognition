# Handwritten Alphabet Recognition System (A-Z)

## Project Overview
This project implements a Deep Neural Network (DNN) to classify handwritten English alphabets (A-Z). The system achieves **99.05% accuracy** on the test set.

## Tech Stack
- **Framework:** Keras / TensorFlow
- **Core Libraries:** NumPy, Pandas, Matplotlib
- **Machine Learning:** Scikit-learn (Data splitting)
- **Dataset:** A-Z Handwritten Alphabets (CSV format, 28x28 grayscale images)

## Key Technical Features
- **Architecture:** Sequential model with Dense layers (512 units) and ReLU activation.
- **Data Preprocessing:** Implemented custom normalization and One-Hot Encoding for 26 classes.
- **Optimization:** Optimized using the **Adam** optimizer with Categorical Crossentropy loss function.
- **Performance:** Achieved high convergence within 15 epochs without overfitting.

## Results
The model shows excellent stability between training and validation metrics.
![Accuracy Graph](images/accuracy_plot.png)
![Loss Graph](images/loss_plot.png)

## How to Run
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run `python model_training.py`.
