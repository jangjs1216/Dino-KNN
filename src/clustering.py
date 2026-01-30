import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class PatchClusterer:
    def __init__(self, n_clusters=20, n_components=64):
        self.n_clusters = n_clusters
        self.n_components = n_components
        
        # Pipelines for reduction and clustering
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    def fit_predict(self, features):
        """
        features: Numpy array (N_patches, dim)
        returns: labels (N_patches,)
        """
        print(f"Standardizing features... shape {features.shape}")
        features_norm = self.scaler.fit_transform(features)
        
        print(f"Running PCA (dim={self.n_components})...")
        features_reduced = self.pca.fit_transform(features_norm)
        print(f"Explained Variance: {np.sum(self.pca.explained_variance_ratio_):.2f}")
        
        print(f"Running K-Means (k={self.n_clusters})...")
        labels = self.kmeans.fit_predict(features_reduced)
        
        return labels, features_reduced
