import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class RegionalClusterAnalyzer:
    """
    Cross-validation of clustering across different regions.
    """

    def __init__(self, df, feature_cols):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.regional_results = {}

    def cluster_by_region(self, k=4, log_transform_cols=None):
        """Perform clustering separately for each region."""
        log_transform_cols = log_transform_cols or ['views', 'likes', 'dislikes', 'comment_count']
        
        regions = self.df['region'].unique()
        print(f"Found {len(regions)} regions: {list(regions)}")
        
        for region in regions:
            region_df = self.df[self.df['region'] == region].copy()
            
            if len(region_df) < k * 10:
                print(f"Skipping {region}: insufficient data ({len(region_df)} samples)")
                continue
            
            
            X = region_df[self.feature_cols].copy()
            
            
            model_features = []
            for col in self.feature_cols:
                if col in log_transform_cols:
                    X[f'log_{col}'] = np.log1p(X[col])
                    model_features.append(f'log_{col}')
                else:
                    model_features.append(col)
            
            
            X_scaled = self.scaler.fit_transform(X[model_features])
            
            
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            
            sil_score = silhouette_score(X_scaled, labels)
            
            
            self.regional_results[region] = {
                'n_samples': len(region_df),
                'labels': labels,
                'silhouette': sil_score,
                'centroids': kmeans.cluster_centers_,
                'X_scaled': X_scaled,
                'df': region_df,
                'model': kmeans
            }
            
            print(f"Region {region}: {len(region_df)} samples, silhouette={sil_score:.3f}")
        
        return self.regional_results

    def compare_regional_clusters(self):
        """Compare clustering results across regions."""
        if not self.regional_results:
            self.cluster_by_region()
        
        comparison = []
        for region, result in self.regional_results.items():
            labels = result['labels']
            unique, counts = np.unique(labels, return_counts=True)
            
            comparison.append({
                'Region': region,
                'N_Samples': result['n_samples'],
                'Silhouette': result['silhouette'],
                **{f'Cluster_{i}_pct': 100 * counts[i] / len(labels) if i < len(counts) else 0 
                   for i in range(max(len(c) for c in [np.unique(r['labels']) for r in self.regional_results.values()]))}
            })
        
        return pd.DataFrame(comparison)

    def cross_region_stability(self, reference_region=None):
        """Measure how well clusters from one region predict another."""
        if not self.regional_results:
            self.cluster_by_region()
        
        regions = list(self.regional_results.keys())
        
        if reference_region is None:
            
            reference_region = max(regions, key=lambda r: self.regional_results[r]['n_samples'])
        
        print(f"Using {reference_region} as reference region")
        
        ref_model = self.regional_results[reference_region]['model']
        ref_scaler = StandardScaler()
        ref_scaler.fit(self.regional_results[reference_region]['X_scaled'])
        
        stability_results = []
        
        for region in regions:
            if region == reference_region:
                continue
            
            
            X_scaled = self.regional_results[region]['X_scaled']
            original_labels = self.regional_results[region]['labels']
            
            predicted_labels = ref_model.predict(X_scaled)
            
            
            ari = adjusted_rand_score(original_labels, predicted_labels)
            nmi = normalized_mutual_info_score(original_labels, predicted_labels)
            
            stability_results.append({
                'Region': region,
                'ARI': ari,
                'NMI': nmi,
                'N_Samples': len(original_labels)
            })
        
        return pd.DataFrame(stability_results)

    def plot_regional_comparison(self):
        """Visualize regional clustering comparison."""
        if not self.regional_results:
            self.cluster_by_region()
        
        comparison_df = self.compare_regional_clusters()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        
        ax1 = axes[0, 0]
        regions = comparison_df['Region'].tolist()
        silhouettes = comparison_df['Silhouette'].tolist()
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(regions)))
        bars = ax1.bar(regions, silhouettes, color=colors)
        ax1.axhline(y=np.mean(silhouettes), color='r', linestyle='--', label='Mean')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Clustering Quality by Region')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        
        ax2 = axes[0, 1]
        sizes = comparison_df['N_Samples'].tolist()
        ax2.bar(regions, sizes, color=colors)
        ax2.set_ylabel('Number of Videos')
        ax2.set_title('Dataset Size by Region')
        ax2.tick_params(axis='x', rotation=45)
        
        
        ax3 = axes[1, 0]
        cluster_cols = [col for col in comparison_df.columns if col.startswith('Cluster_')]
        if cluster_cols:
            cluster_data = comparison_df[['Region'] + cluster_cols].set_index('Region')
            sns.heatmap(cluster_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3)
            ax3.set_title('Cluster Distribution by Region (%)')
        
        
        ax4 = axes[1, 1]
        stability_df = self.cross_region_stability()
        if not stability_df.empty:
            x = range(len(stability_df))
            width = 0.35
            ax4.bar([i - width/2 for i in x], stability_df['ARI'], width, label='ARI', alpha=0.8)
            ax4.bar([i + width/2 for i in x], stability_df['NMI'], width, label='NMI', alpha=0.8)
            ax4.set_xticks(x)
            ax4.set_xticklabels(stability_df['Region'], rotation=45)
            ax4.set_ylabel('Score')
            ax4.set_title('Cross-Region Cluster Stability')
            ax4.legend()
            ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df

    def plot_regional_cluster_profiles(self, top_regions=4):
        """Compare cluster profiles across top regions."""
        if not self.regional_results:
            self.cluster_by_region()
        
        
        sorted_regions = sorted(self.regional_results.keys(), 
                               key=lambda r: self.regional_results[r]['n_samples'], 
                               reverse=True)[:top_regions]
        
        n_features = len(self.feature_cols)
        fig, axes = plt.subplots(len(sorted_regions), 1, figsize=(14, 4*len(sorted_regions)))
        
        if len(sorted_regions) == 1:
            axes = [axes]
        
        for idx, region in enumerate(sorted_regions):
            ax = axes[idx]
            centroids = self.regional_results[region]['centroids']
            
            x = np.arange(centroids.shape[1])
            width = 0.8 / centroids.shape[0]
            
            for i in range(centroids.shape[0]):
                ax.bar(x + i * width, centroids[i], width, label=f'Cluster {i}', alpha=0.8)
            
            ax.set_xticks(x + width * (centroids.shape[0] - 1) / 2)
            ax.set_xticklabels([f.replace('log_', '') for f in self.feature_cols], rotation=45)
            ax.set_ylabel('Normalized Value')
            ax.set_title(f'{region} - Cluster Centroids (n={self.regional_results[region]["n_samples"]})')
            ax.legend(loc='upper right')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def find_universal_patterns(self, k=4):
        """Identify patterns that are consistent across regions."""
        if not self.regional_results:
            self.cluster_by_region(k=k)
        
        
        all_centroids = []
        centroid_regions = []
        
        for region, result in self.regional_results.items():
            for i, centroid in enumerate(result['centroids']):
                all_centroids.append(centroid)
                centroid_regions.append(f'{region}_C{i}')
        
        all_centroids = np.array(all_centroids)
        
        
        meta_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        meta_labels = meta_kmeans.fit_predict(all_centroids)
        
        
        pattern_mapping = pd.DataFrame({
            'region_cluster': centroid_regions,
            'universal_pattern': meta_labels
        })
        
        print("\nUniversal Pattern Mapping:")
        for pattern in range(k):
            members = pattern_mapping[pattern_mapping['universal_pattern'] == pattern]['region_cluster'].tolist()
            print(f"  Pattern {pattern}: {members}")
        
        return pattern_mapping, meta_kmeans.cluster_centers_
