# 🌲 Forest Cover Type Prediction using Random Forest

This project focuses on predicting the forest cover type of land areas using supervised machine learning. The model uses the **Random Forest Classifier** to predict cover types based on various cartographic variables.

## 📌 Problem Statement

The goal is to predict the type of forest cover for a given plot of land using cartographic features such as elevation, slope, soil type, etc. This is a multiclass classification problem.

## 📊 Dataset

- **Source**: [Roosevelt National Forest of northern Colorado]
- **Size**: 15000+ rows, 50+ columns
- **Target Variable**: `Cover_Type` (5 classes)
- **Features Include**:
  - Elevation, Aspect, Slope
  - Horizontal & Vertical Distance to Hydrology
  - Soil type and wilderness area indicators

## 🔧 Technologies & Libraries Used

- Python 3.x
- `pandas`, `numpy` for data handling
- `scikit-learn` for model building and evaluation
- Feature engineering
- Jupyter Notebook
- Pickle for future loading of pipeline

## 📈 Model and Evaluation

- **Algorithm Used**: Random Forest Classifier
- **Train-Test Split**: 80/20
- **Metrics**:
  - Precision (0.843), Recall(0.8458), F1-score per class(0.8437)

### ✅ Why Random Forest?
- Handles high-dimensional datasets well
- Reduces overfitting compared to individual decision trees
- Works well with non-linear features
