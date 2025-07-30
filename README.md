# 🎤 Human Voice Classification and Clustering

This project classifies and clusters human voice samples using extracted audio features. It supports model training, evaluation, and visualization through a user-friendly Streamlit interface.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Features](#features)
- [Models Used](#models-used)
- [How to Run](#how-to-run)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Technologies Used](#technologies-used)
- [Author](#author)

---

## 🔍 Overview

The goal is to build a machine learning pipeline that:

- Preprocesses extracted voice features
- Classifies voice samples as male or female
- Clusters similar voice samples using KMeans and DBSCAN
- Evaluates and compares models
- Provides a Streamlit UI for predictions and EDA

---

## ⚙️ Project Architecture

```
📦Human_Voice_Clustering_Project/
│
├── app.py                     # Streamlit app
├── train.py                  # Model training and evaluation
├── data/
│   └── vocal_gender_features_new.csv
├── models/                   # Saved classification models
├── src/
│   ├── classification.py     # Training, evaluation, confusion matrix
│   ├── clustering.py         # KMeans and DBSCAN functions
│   ├── eda.py                # Visualizations (EDA and feature importance)
│   ├── preprocess.py         # Cleaning, scaling, feature extraction
├── README.md
└── requirements.txt
```

---

## 🌟 Features

- ✅ Data Cleaning & Normalization
- ✅ Random Forest, SVM, MLP Classifiers
- ✅ Classification Reports & Accuracy Scores
- ✅ Confusion Matrix Visualization
- ✅ Feature Importance (for RF)
- ✅ KMeans Clustering Evaluation (Silhouette & Purity)
- ✅ DBSCAN Clustering with tuning
- ✅ Label Distribution and Correlation Heatmap
- ✅ Interactive Streamlit UI

---

## 🤖 Models Used

| Model         | Type       | Purpose                |
| ------------- | ---------- | ---------------------- |
| Random Forest | Classifier | Gender prediction      |
| SVM           | Classifier | Gender prediction      |
| MLP           | Classifier | Gender prediction      |
| KMeans        | Clustering | Group similar voices   |
| DBSCAN        | Clustering | Density-based grouping |

---

## 🚀 How to Run

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train models**

   ```bash
   python train.py
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

4. **Upload CSV file** (`vocal_gender_features_new.csv`) in the app to test and explore.

---

## 🖼️ Screenshots

- ✅ Gender Predictions Table
- 📊 Confusion Matrix
- 📈 Top Feature Importance (Random Forest)
- 🔍 Correlation Heatmap
- 📉 Label Distribution

---

## 📁 Project Structure

- `train.py`: Model training and evaluation (Random Forest, SVM, MLP, KMeans, DBSCAN)
- `app.py`: Interactive Streamlit app for model selection and EDA
- `src/classification.py`: Training, tuning, evaluation, confusion matrix
- `src/clustering.py`: KMeans and DBSCAN clustering logic
- `src/eda.py`: Label distribution, correlation heatmap, feature importance
- `src/preprocess.py`: Data cleaning, feature extraction, scaling

---

## 🧪 Future Work

- [ ] Live audio input using `librosa` for real-time prediction
- [ ] Exportable comparison reports

---

## 🛠 Technologies Used

- Python
- Scikit-learn
- Streamlit
- Matplotlib, Seaborn
- Pandas, NumPy
- Machine Learning (Classification & Clustering)
