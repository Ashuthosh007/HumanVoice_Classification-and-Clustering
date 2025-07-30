import streamlit as st
import pandas as pd
import joblib
from src.preprocess import preprocess_features, clean_data
from src.classification import evaluate_model, show_confusion_matrix
import src.eda as eda
from src.clustering import kmeans_clustering, dbscan_clustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


st.title("🎤 Human Voice Classification")

uploaded_file = st.file_uploader("Upload CSV with voice features", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = clean_data(df)
    X, y, _ = preprocess_features(df)
    X = pd.DataFrame(X, columns=df.drop("label", axis=1).columns)


    model_choice = st.selectbox("Choose a classification model", ["Random Forest", "SVM", "MLP"])
    model_path = f"models/{model_choice.replace(' ', '_').lower()}_model.pkl"

    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {model_path}. Please train the model first using `train.py`.")
        st.stop()
    preds = model.predict(X)

    df["Predicted Label"] = preds
    df["Gender"] = df["Predicted Label"].map({0: "Female", 1: "Male"})

    st.success("✅ Prediction Complete")
    st.write(df[["Predicted Label", "Gender"]])

    if "label" in df.columns:
        st.subheader("📊 Confusion Matrix")
        show_confusion_matrix(df["label"], df["Predicted Label"])

    # Show Feature Importance (only for models like RandomForest)
    if st.checkbox("Show Feature Importances"):
        st.subheader("📈 Top Feature Importances")
        eda.show_feature_importance(model, feature_names=X.columns.tolist())


# EDA Section
st.subheader("🔍 Exploratory Data Analysis")

if st.checkbox("Show Label Distribution"):
    eda.label_distribution(df)

if st.checkbox("Show Feature Correlation Heatmap"):
    eda.feature_correlation(df)

# Clustering Section
st.subheader("🔍 Clustering Visualization (PCA 2D)")

if st.checkbox("Show KMeans Clusters"):
    kmeans_model, kmeans_labels, _ = kmeans_clustering(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    cluster_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    cluster_df["Cluster"] = kmeans_labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=cluster_df, palette="Set2")
    plt.title("KMeans Clustering (PCA 2D)")
    st.pyplot(plt.gcf())
    plt.clf()
