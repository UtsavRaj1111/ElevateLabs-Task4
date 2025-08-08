# Logistic Regression - Binary Classification Project

## 📌 Overview
This project demonstrates how to build a binary classification model using Logistic Regression in Python.  
The model is trained on the Breast Cancer Wisconsin dataset, which is included in `scikit-learn`.  
The workflow covers data preprocessing, model training, evaluation, threshold tuning, and sigmoid function analysis.

The primary goal is to classify tumors as malignant (cancerous) or benign (non-cancerous) based on various medical features.

---

## 🎯 Objectives
1. Load and explore a binary classification dataset.
2. Perform train/test split and feature standardization.
3. Train a Logistic Regression model.
4. Evaluate the model using:
   - Confusion Matrix
   - Precision
   - Recall
   - ROC-AUC Score
5. Tune the classification threshold to improve performance.
6. Explain and visualize the sigmoid function used in logistic regression.

---

## 📂 Dataset
- Name: Breast Cancer Wisconsin (Diagnostic) Dataset  
- Source: Built-in dataset from `scikit-learn`  
- Target Variable: 
  - `0` → Malignant  
  - `1` → Benign  
- Features: 30 numeric medical measurements such as radius, texture, smoothness, and symmetry.

---

## 🛠️ Technologies Used
- Python 3.x
- pandas – Data handling
- numpy – Numerical operations
- matplotlib – Data visualization
- scikit-learn – Machine learning tools and dataset

---

## 📊 Project Workflow
1. Data Loading 
   Load dataset directly from `scikit-learn` without downloading files.
   
2. Data Preprocessing  
   - Split data into training and test sets (`train_test_split`).
   - Standardize features using `StandardScaler` for better model performance.

3. Model Training
   Train a `LogisticRegression` model on the standardized features.

4. Evaluation Metrics 
   - Confusion Matrix to analyze prediction distribution.  
   - Precision to measure how many predicted positives are correct.  
   - Recall to measure how many actual positives were correctly identified.  
   - ROC-AUC to evaluate classification performance across thresholds.

5. ROC Curve Plotting
   Visual representation of the trade-off between true positive rate and false positive rate.

6. Threshold Tuning
   Adjust classification threshold from the default 0.5 to improve recall or precision depending on the use case.

7. Sigmoid Function Visualization 
   Plot the sigmoid curve to show how logistic regression maps input values to probabilities.

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/logistic-regression-project.git
cd logistic-regression-project
