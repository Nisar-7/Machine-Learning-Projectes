# Machine-Learning-Projectes
Implementation of various ML models (KNN, SVM, RF, DT, LR, NB, ANN, PCA etc.) with performance evaluation (Accuracy, Precision, Recall, F1-Score) and result visualization using Matplotlib. Includes data preprocessing, training, testing, and comparison.

# Breast Cancer Classification Using Various Machine Learning Algorithms

This project focuses on classifying breast cancer data using multiple machine learning models. The dataset used is the well-known Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn. The goal is to compare the performance of different classifiers on this real-world medical dataset.

---

## Dataset
- **Breast Cancer Wisconsin (Diagnostic)** dataset from scikit-learn
- Contains features computed from digitized images of breast mass tissue samples
- Target labels: malignant or benign tumors

---

## Project Overview

1. **Data Preprocessing:**
   - Split dataset into training (80%) and testing (20%) sets.
   - Normalize feature values to the [0, 1] range using `MinMaxScaler`.

2. **Models Implemented:**
   - Gaussian Naive Bayes (GNB)
   - K-Nearest Neighbors (KNN)
   - Decision Tree (DT)
   - Random Forest (RF)
   - Support Vector Machine (SVM) with polynomial kernel
   - Logistic Regression (LR)
   - Artificial Neural Network (ANN) using Multi-layer Perceptron

3. **Performance Metrics:**
   - Accuracy
   - Precision
   - Recall

4. **Evaluation:**
   - Train each model on training data.
   - Predict and evaluate on both training and testing data.
   - Visualize training accuracies of all models for comparison.

---

## How to Use

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install scikit-learn matplotlib numpy
