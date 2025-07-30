from src.preprocess import load_data, clean_data, preprocess_features, split_data
from src.classification import train_model, evaluate_model
from src.clustering import kmeans_clustering, dbscan_clustering
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import os
import pandas as pd
import numpy as np

# Function to compute cluster purity
def cluster_purity(y_true, y_pred):
    contingency_matrix = pd.crosstab(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)

# Load and process
df = load_data("data/vocal_gender_features_new.csv")
df = clean_data(df)
X, y, scaler = preprocess_features(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Create models
models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "MLP": MLPClassifier(max_iter=500)
}

# Train and evaluate each model
os.makedirs("models", exist_ok=True)
for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model = train_model(model, X_train, y_train)
    acc, report = evaluate_model(model, X_test, y_test)

    print(f"=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(report)

    # Save model
    model_filename = f"models/{name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, model_filename)

# Save scaler (only once)
joblib.dump(scaler, "models/scaler.pkl")

# --- Clustering Evaluation ---

# 1. KMeans on full features
kmeans_model, kmeans_labels, kmeans_silhouette = kmeans_clustering(X)
kmeans_purity = cluster_purity(y, kmeans_labels)

print("\n=== KMeans Clustering Evaluation ===")
print(f"Silhouette Score: {kmeans_silhouette:.4f}")
print(f"Cluster Purity: {kmeans_purity:.4f}")

# 2. DBSCAN after PCA dimensionality reduction
print("\n=== DBSCAN Clustering Evaluation ===")
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

for eps in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    for min_samples in [3, 5, 10]:
        print(f"\n--- DBSCAN (eps={eps}, min_samples={min_samples}) ---")
        db_model, db_labels, db_sil = dbscan_clustering(X_pca, eps=eps, min_samples=min_samples)
        if np.any(db_labels != -1):
            purity = cluster_purity(y[db_labels != -1], db_labels[db_labels != -1])
            print(f"Silhouette Score: {db_sil:.4f}")
            print(f"Cluster Purity: {purity:.4f}")
        else:
            print("All points marked as noise.")

