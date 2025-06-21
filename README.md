# 🌲 Forest Cover Type Prediction using Random Forest

This project is part of my AI/ML internship assignment and focuses on predicting the forest cover type of land areas using supervised machine learning. The model uses the **Random Forest Classifier** to predict cover types based on various cartographic variables.

## 📌 Problem Statement

The goal is to predict the type of forest cover for a given plot of land using cartographic features such as elevation, slope, soil type, etc. This is a multiclass classification problem.

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository - Forest CoverType Dataset](https://archive.ics.uci.edu/ml/datasets/Covertype)
- **Size**: ~580,000 rows, 55 columns
- **Target Variable**: `Cover_Type` (7 classes)
- **Features Include**:
  - Elevation, Aspect, Slope
  - Horizontal & Vertical Distance to Hydrology
  - Soil type and wilderness area indicators

## 🔧 Technologies & Libraries Used

- Python 3.x
- `pandas`, `numpy` for data handling
- `matplotlib`, `seaborn` for visualization
- `scikit-learn` for model building and evaluation
- Jupyter Notebook

## 📈 Model and Evaluation

- **Algorithm Used**: Random Forest Classifier
- **Train-Test Split**: 80/20
- **Metrics**:
  - Accuracy: `XX.XX%`
  - Precision, Recall, F1-score per class
  - Confusion Matrix

### ✅ Why Random Forest?
- Handles high-dimensional datasets well
- Reduces overfitting compared to individual decision trees
- Works well with non-linear features

## 🔍 Results

| Metric | Value |
| Accuracy | XX.XX% |
