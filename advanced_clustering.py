import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings

warnings.filterwarnings('ignore')


class AdvancedClusterEngine:
    """
    Advanced clustering with multiple algorithms and evaluation metrics.
    """

    def __init__(self, df, feature_cols=None):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.feature_cols = feature_cols or ['views', 'likes', 'dislikes', 'comment_count', 'days_to_trend', 'title_length']
        self.X_scaled = None
        self.cluster_results = {}
        
    def prepare_features(self, log_transform_cols=None):
        """Prepare and scale features for clustering."""
        log_transform_cols = log_transform_cols or ['views', 'likes', 'dislikes', 'comment_count']
        
        self.X = self.df[self.feature_cols].copy()
        
        
        self.model_features = []
        for col in self.feature_cols:
            if col in log_transform_cols:
                self.X[f'log_{col}'] = np.log1p(self.X[col])
                self.model_features.append(f'log_{col}')
            else:
                self.model_features.append(col)
        
        self.X_scaled = self.scaler.fit_transform(self.X[self.model_features])
        return self.model_features

    def kmeans_clustering(self, k=4):
        """Standard K-Means clustering."""
        model = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        labels = model.fit_predict(self.X_scaled)
        self.cluster_results['kmeans'] = {
            'model': model,
            'labels': labels,
            'n_clusters': k
        }
        return labels

    def dbscan_clustering(self, eps=0.5, min_samples=5):
        """DBSCAN density-based clustering."""
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(self.X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.cluster_results['dbscan'] = {
            'model': model,
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': list(labels).count(-1)
        }
        return labels

    def hierarchical_clustering(self, k=4, linkage_method='ward'):
        """Agglomerative Hierarchical clustering."""
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        labels = model.fit_predict(self.X_scaled)
        self.cluster_results['hierarchical'] = {
            'model': model,
            'labels': labels,
            'n_clusters': k,
            'linkage_method': linkage_method
        }
        return labels

    def gmm_clustering(self, k=4):
        """Gaussian Mixture Model clustering."""
        model = GaussianMixture(n_components=k, random_state=42, n_init=5)
        labels = model.fit_predict(self.X_scaled)
        probs = model.predict_proba(self.X_scaled)
        self.cluster_results['gmm'] = {
            'model': model,
            'labels': labels,
            'probabilities': probs,
            'n_clusters': k,
            'bic': model.bic(self.X_scaled),
            'aic': model.aic(self.X_scaled)
        }
        return labels

    def run_all_algorithms(self, k=4, dbscan_eps=0.5, dbscan_min_samples=5):
        """Run all clustering algorithms and store results."""
        print("Running K-Means...")
        self.kmeans_clustering(k)
        
        print("Running DBSCAN...")
        self.dbscan_clustering(dbscan_eps, dbscan_min_samples)
        
        print("Running Hierarchical Clustering...")
        self.hierarchical_clustering(k)
        
        print("Running Gaussian Mixture Model...")
        self.gmm_clustering(k)
        
        return self.cluster_results

    def find_optimal_dbscan_params(self, eps_range=None, min_samples_range=None):
        """Find optimal DBSCAN parameters using silhouette score."""
        eps_range = eps_range or np.arange(0.3, 1.5, 0.1)
        min_samples_range = min_samples_range or range(3, 10)
        
        best_score = -1
        best_params = {}
        results = []
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.X_scaled)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                if n_clusters > 1 and n_noise < len(labels) * 0.5:
                    mask = labels != -1
                    if mask.sum() > n_clusters:
                        score = silhouette_score(self.X_scaled[mask], labels[mask])
                        results.append({
                            'eps': eps, 'min_samples': min_samples,
                            'n_clusters': n_clusters, 'n_noise': n_noise,
                            'silhouette': score
                        })
                        if score > best_score:
                            best_score = score
                            best_params = {'eps': eps, 'min_samples': min_samples}
        
        return best_params, pd.DataFrame(results)

    def compare_algorithms(self):
        """Compare all algorithms using multiple metrics."""
        comparison = []
        
        for name, result in self.cluster_results.items():
            labels = result['labels']
            
            
            valid_mask = labels != -1
            n_valid = valid_mask.sum()
            n_clusters = len(set(labels[valid_mask]))
            
            if n_clusters < 2 or n_valid < n_clusters + 1:
                comparison.append({
                    'Algorithm': name.upper(),
                    'N_Clusters': result['n_clusters'],
                    'Silhouette': np.nan,
                    'Calinski-Harabasz': np.nan,
                    'Davies-Bouldin': np.nan
                })
                continue
            
            X_valid = self.X_scaled[valid_mask]
            labels_valid = labels[valid_mask]
            
            metrics = {
                'Algorithm': name.upper(),
                'N_Clusters': n_clusters,
                'Silhouette': silhouette_score(X_valid, labels_valid),
                'Calinski-Harabasz': calinski_harabasz_score(X_valid, labels_valid),
                'Davies-Bouldin': davies_bouldin_score(X_valid, labels_valid)
            }
            
            if name == 'gmm':
                metrics['BIC'] = result['bic']
                metrics['AIC'] = result['aic']
            
            if name == 'dbscan':
                metrics['N_Noise'] = result['n_noise']
            
            comparison.append(metrics)
        
        return pd.DataFrame(comparison)

    def plot_algorithm_comparison(self):
        """Visualize comparison of clustering algorithms."""
        comparison_df = self.compare_algorithms()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        
        ax1 = axes[0, 0]
        valid_sil = comparison_df.dropna(subset=['Silhouette'])
        if not valid_sil.empty:
            colors = sns.color_palette('viridis', len(valid_sil))
            bars = ax1.bar(valid_sil['Algorithm'], valid_sil['Silhouette'], color=colors)
            ax1.set_title('Silhouette Score by Algorithm (Higher is Better)')
            ax1.set_ylabel('Silhouette Score')
            ax1.axhline(y=0.5, color='r', linestyle='--', label='Good threshold')
            ax1.legend()
        
        
        ax2 = axes[0, 1]
        valid_ch = comparison_df.dropna(subset=['Calinski-Harabasz'])
        if not valid_ch.empty:
            colors = sns.color_palette('viridis', len(valid_ch))
            ax2.bar(valid_ch['Algorithm'], valid_ch['Calinski-Harabasz'], color=colors)
            ax2.set_title('Calinski-Harabasz Index (Higher is Better)')
            ax2.set_ylabel('CH Index')
        
        
        ax3 = axes[1, 0]
        valid_db = comparison_df.dropna(subset=['Davies-Bouldin'])
        if not valid_db.empty:
            colors = sns.color_palette('viridis', len(valid_db))
            ax3.bar(valid_db['Algorithm'], valid_db['Davies-Bouldin'], color=colors)
            ax3.set_title('Davies-Bouldin Index (Lower is Better)')
            ax3.set_ylabel('DB Index')
        
        
        ax4 = axes[1, 1]
        cluster_counts = []
        for name, result in self.cluster_results.items():
            labels = result['labels']
            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                cluster_counts.append({'Algorithm': name.upper(), 'Cluster': str(u), 'Count': c})
        
        if cluster_counts:
            count_df = pd.DataFrame(cluster_counts)
            count_pivot = count_df.pivot(index='Algorithm', columns='Cluster', values='Count').fillna(0)
            count_pivot.plot(kind='bar', stacked=True, ax=ax4, colormap='viridis')
            ax4.set_title('Cluster Size Distribution by Algorithm')
            ax4.set_ylabel('Number of Samples')
            ax4.legend(title='Cluster', bbox_to_anchor=(1.02, 1))
            ax4.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df

    def plot_dendrogram(self, max_display=50):
        """Plot hierarchical clustering dendrogram."""
        
        if len(self.X_scaled) > 1000:
            sample_idx = np.random.choice(len(self.X_scaled), 1000, replace=False)
            X_sample = self.X_scaled[sample_idx]
        else:
            X_sample = self.X_scaled
        
        plt.figure(figsize=(14, 8))
        
        linkage_matrix = linkage(X_sample, method='ward')
        
        dendrogram(
            linkage_matrix,
            truncate_mode='lastp',
            p=max_display,
            leaf_rotation=90,
            leaf_font_size=8,
            show_contracted=True
        )
        
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index (or Cluster Size)')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.show()

    def plot_gmm_probabilities(self):
        """Visualize GMM cluster assignment probabilities."""
        if 'gmm' not in self.cluster_results:
            print("Run GMM clustering first!")
            return
        
        probs = self.cluster_results['gmm']['probabilities']
        labels = self.cluster_results['gmm']['labels']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        
        ax1 = axes[0]
        max_probs = np.max(probs, axis=1)
        ax1.hist(max_probs, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0.5, color='r', linestyle='--', label='50% threshold')
        ax1.axvline(x=0.8, color='g', linestyle='--', label='80% threshold')
        ax1.set_title('Distribution of Maximum Cluster Probabilities')
        ax1.set_xlabel('Maximum Probability')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        
        ax2 = axes[1]
        sample_size = min(100, len(probs))
        sample_idx = np.random.choice(len(probs), sample_size, replace=False)
        sample_probs = probs[sample_idx]
        
        sns.heatmap(sample_probs, cmap='YlOrRd', ax=ax2, 
                    xticklabels=[f'Cluster {i}' for i in range(probs.shape[1])],
                    yticklabels=False)
        ax2.set_title(f'Cluster Probabilities (Sample of {sample_size})')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Samples')
        
        plt.tight_layout()
        plt.show()
        
        
        uncertain = np.sum(max_probs < 0.5)
        confident = np.sum(max_probs >= 0.8)
        print(f"\nGMM Assignment Confidence:")
        print(f"  Uncertain assignments (max prob < 50%): {uncertain} ({100*uncertain/len(probs):.1f}%)")
        print(f"  Confident assignments (max prob >= 80%): {confident} ({100*confident/len(probs):.1f}%)")
