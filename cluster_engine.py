import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class ClusterEngine:
    """
    Performs K-Means clustering and visualization.
    """
    def __init__(self, df):
        self.df = df.copy()
        self.model = None
        self.scaler = StandardScaler()

    def prepare_features(self):
        features = ['views', 'likes', 'dislikes', 'comment_count', 'days_to_trend', 'title_length']
        self.X = self.df[features].copy()

        for col in ['views', 'likes', 'dislikes', 'comment_count']:
            self.X[f'log_{col}'] = np.log1p(self.X[col])

        model_features = ['log_views', 'log_likes', 'log_dislikes', 'log_comment_count', 'days_to_trend',
                          'title_length']
        self.X_scaled = self.scaler.fit_transform(self.X[model_features])
        return model_features

    def find_optimal_k(self, max_k=10):
        print("Calculating Elbow Curve...")
        wcss = []
        for i in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(self.X_scaled)
            wcss.append(kmeans.inertia_)

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
        plt.title('Elbow Method to Determine Optimal Clusters')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

    def apply_clustering(self, k=4):
        print(f"Applying K-Means with k={k}...")
        self.model = KMeans(n_clusters=k, init='k-means++', random_state=42)
        clusters = self.model.fit_predict(self.X_scaled)
        self.df['Cluster'] = clusters
        return self.df

    def visualize_clusters_pca(self):
        """Reduces dimensions to 2D using PCA to visualize cluster separation."""
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)

        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=self.df['Cluster'], palette='deep', alpha=0.6, s=50)
        plt.title("Cluster Visualization using PCA (2D Projection)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title='Cluster')
        plt.show()

    def get_cluster_stats(self):
        """Returns a summary of metrics per cluster for analysts."""

        metrics = ['views', 'likes', 'dislikes', 'comment_count', 'days_to_trend', 'engagement_rate', 'title_length']
        summary = self.df.groupby('Cluster')[metrics].mean().reset_index()

        counts = self.df['Cluster'].value_counts().reset_index()
        counts.columns = ['Cluster', 'Count']

        summary = pd.merge(summary, counts, on='Cluster')
        return summary
