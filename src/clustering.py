# src/clustering.py
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

def kmeans_clustering(X, n_clusters=2):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    silhouette = silhouette_score(X, labels)
    return model, labels, silhouette

def dbscan_clustering(X, eps=1.0, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    # Filter noise (-1) for silhouette score
    core_mask = labels != -1
    if np.sum(core_mask) > 1 and len(np.unique(labels[core_mask])) > 1:
        silhouette = silhouette_score(X[core_mask], labels[core_mask])
    else:
        silhouette = -1.0  # Not enough clusters
    return model, labels, silhouette
