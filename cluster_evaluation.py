import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.utils import resample
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class ClusterEvaluator:
    """
    Comprehensive cluster evaluation with silhouette analysis and stability testing.
    """

    def __init__(self, X_scaled, feature_names=None):
        self.X_scaled = X_scaled
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X_scaled.shape[1])]

    def silhouette_analysis(self, k_range=None):
        """Perform silhouette analysis for different k values."""
        k_range = k_range or range(2, 11)
        results = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_scaled)
            
            sil_avg = silhouette_score(self.X_scaled, labels)
            sil_samples = silhouette_samples(self.X_scaled, labels)
            
            
            cluster_sils = {}
            for i in range(k):
                cluster_mask = labels == i
                cluster_sils[i] = sil_samples[cluster_mask].mean()
            
            results.append({
                'k': k,
                'silhouette_avg': sil_avg,
                'silhouette_std': sil_samples.std(),
                'cluster_silhouettes': cluster_sils,
                'labels': labels,
                'silhouette_samples': sil_samples
            })
        
        return results

    def plot_silhouette_analysis(self, k_range=None):
        """Visualize silhouette analysis results."""
        results = self.silhouette_analysis(k_range)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        
        ax1 = axes[0, 0]
        ks = [r['k'] for r in results]
        sils = [r['silhouette_avg'] for r in results]
        ax1.plot(ks, sils, 'bo-', linewidth=2, markersize=8)
        ax1.fill_between(ks, 
                         [r['silhouette_avg'] - r['silhouette_std'] for r in results],
                         [r['silhouette_avg'] + r['silhouette_std'] for r in results],
                         alpha=0.2)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Average Silhouette Score')
        ax1.set_title('Silhouette Score vs Number of Clusters')
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Good threshold')
        ax1.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Fair threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        
        best_idx = np.argmax(sils)
        best_result = results[best_idx]
        
        ax2 = axes[0, 1]
        self._plot_silhouette_diagram(ax2, best_result)
        ax2.set_title(f'Silhouette Plot for Best k={best_result["k"]}')
        
        
        ax3 = axes[1, 0]
        cluster_sil_matrix = []
        for r in results:
            row = [r['cluster_silhouettes'].get(i, 0) for i in range(r['k'])]
            row.extend([np.nan] * (max(ks) - len(row)))
            cluster_sil_matrix.append(row)
        
        cluster_sil_df = pd.DataFrame(
            cluster_sil_matrix,
            index=[f'k={k}' for k in ks],
            columns=[f'C{i}' for i in range(max(ks))]
        )
        sns.heatmap(cluster_sil_df, annot=True, fmt='.2f', cmap='RdYlGn', 
                    center=0.5, ax=ax3, mask=cluster_sil_df.isna())
        ax3.set_title('Per-Cluster Silhouette Scores')
        
        
        ax4 = axes[1, 1]
        ax4.hist(best_result['silhouette_samples'], bins=50, edgecolor='black', alpha=0.7)
        ax4.axvline(x=best_result['silhouette_avg'], color='r', linestyle='--', 
                    label=f'Mean: {best_result["silhouette_avg"]:.3f}')
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('Silhouette Coefficient')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Silhouette Distribution (k={best_result["k"]})')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
        return results

    def _plot_silhouette_diagram(self, ax, result):
        """Create a silhouette diagram for a specific clustering result."""
        labels = result['labels']
        sil_samples = result['silhouette_samples']
        k = result['k']
        
        y_lower = 10
        colors = plt.cm.viridis(np.linspace(0, 1, k))
        
        for i in range(k):
            cluster_sils = sil_samples[labels == i]
            cluster_sils.sort()
            
            size = cluster_sils.shape[0]
            y_upper = y_lower + size
            
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sils,
                            facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size, str(i))
            
            y_lower = y_upper + 10
        
        ax.axvline(x=result['silhouette_avg'], color='red', linestyle='--')
        ax.set_xlabel('Silhouette Coefficient')
        ax.set_ylabel('Cluster')
        ax.set_yticks([])

    def cluster_stability_analysis(self, k=4, n_bootstrap=100, sample_fraction=0.8):
        """
        Assess cluster stability using bootstrap sampling.
        Returns stability scores indicating how consistently points are assigned.
        """
        print(f"Running stability analysis with {n_bootstrap} bootstrap samples...")
        n_samples = len(self.X_scaled)
        sample_size = int(n_samples * sample_fraction)
        
        
        assignment_matrix = np.zeros((n_samples, n_bootstrap), dtype=int)
        inclusion_count = np.zeros(n_samples, dtype=int)
        
        for b in range(n_bootstrap):
            
            indices = resample(range(n_samples), n_samples=sample_size, random_state=b)
            X_boot = self.X_scaled[indices]
            
            
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            kmeans.fit(X_boot)
            
            
            all_labels = kmeans.predict(self.X_scaled)
            assignment_matrix[:, b] = all_labels
            
            
            for idx in indices:
                inclusion_count[idx] += 1
        
        
        stability_scores = []
        for i in range(n_samples):
            if inclusion_count[i] > 0:
                assignments = assignment_matrix[i, :]
                most_common = Counter(assignments).most_common(1)[0][1]
                stability = most_common / n_bootstrap
                stability_scores.append(stability)
            else:
                stability_scores.append(0)
        
        stability_scores = np.array(stability_scores)
        
        
        final_kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        final_labels = final_kmeans.fit_predict(self.X_scaled)
        
        return {
            'stability_scores': stability_scores,
            'mean_stability': stability_scores.mean(),
            'std_stability': stability_scores.std(),
            'labels': final_labels,
            'assignment_matrix': assignment_matrix,
            'k': k
        }

    def plot_stability_analysis(self, k=4, n_bootstrap=100):
        """Visualize cluster stability results."""
        results = self.cluster_stability_analysis(k, n_bootstrap)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        
        ax1 = axes[0, 0]
        ax1.hist(results['stability_scores'], bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(x=results['mean_stability'], color='r', linestyle='--',
                    label=f'Mean: {results["mean_stability"]:.3f}')
        ax1.set_xlabel('Stability Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Sample Stability Scores')
        ax1.legend()
        
        
        ax2 = axes[0, 1]
        cluster_stabilities = {}
        for c in range(k):
            mask = results['labels'] == c
            cluster_stabilities[c] = results['stability_scores'][mask]
        
        bp = ax2.boxplot([cluster_stabilities[c] for c in range(k)],
                         labels=[f'Cluster {c}' for c in range(k)],
                         patch_artist=True)
        colors = plt.cm.viridis(np.linspace(0, 1, k))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax2.set_ylabel('Stability Score')
        ax2.set_title('Stability Score by Cluster')
        ax2.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='High stability')
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate stability')
        ax2.legend()
        
        
        ax3 = axes[1, 0]
        scatter = ax3.scatter(range(len(results['stability_scores'])), 
                             results['stability_scores'],
                             c=results['labels'], cmap='viridis', alpha=0.5, s=10)
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Stability Score')
        ax3.set_title('Sample Stability Scores (colored by cluster)')
        plt.colorbar(scatter, ax=ax3, label='Cluster')
        
        
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        Cluster Stability Analysis Summary
        ==================================
        
        Number of clusters (k): {k}
        Bootstrap iterations: {n_bootstrap}
        
        Overall Stability:
        • Mean stability score: {results['mean_stability']:.3f}
        • Std stability score: {results['std_stability']:.3f}
        
        Per-Cluster Mean Stability:
        """
        for c in range(k):
            mean_stab = cluster_stabilities[c].mean()
            summary_text += f"\n        • Cluster {c}: {mean_stab:.3f}"
        
        summary_text += f"""
        
        Interpretation:
        • Stability > 0.8: Highly stable clustering
        • Stability 0.5-0.8: Moderately stable
        • Stability < 0.5: Unstable, consider different k
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
        
        return results

    def find_optimal_k_comprehensive(self, k_range=None):
        """Find optimal k using multiple criteria."""
        k_range = k_range or range(2, 11)
        
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
        
        results = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_scaled)
            
            results.append({
                'k': k,
                'silhouette': silhouette_score(self.X_scaled, labels),
                'calinski_harabasz': calinski_harabasz_score(self.X_scaled, labels),
                'davies_bouldin': davies_bouldin_score(self.X_scaled, labels),
                'inertia': kmeans.inertia_
            })
        
        results_df = pd.DataFrame(results)
        
        
        results_df['sil_norm'] = (results_df['silhouette'] - results_df['silhouette'].min()) / \
                                 (results_df['silhouette'].max() - results_df['silhouette'].min())
        results_df['ch_norm'] = (results_df['calinski_harabasz'] - results_df['calinski_harabasz'].min()) / \
                                (results_df['calinski_harabasz'].max() - results_df['calinski_harabasz'].min())
        results_df['db_norm'] = 1 - (results_df['davies_bouldin'] - results_df['davies_bouldin'].min()) / \
                                    (results_df['davies_bouldin'].max() - results_df['davies_bouldin'].min())
        
        
        results_df['combined_score'] = (results_df['sil_norm'] + results_df['ch_norm'] + results_df['db_norm']) / 3
        
        return results_df

    def plot_optimal_k_analysis(self, k_range=None):
        """Visualize comprehensive k selection analysis."""
        results_df = self.find_optimal_k_comprehensive(k_range)
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        
        ax1 = axes[0, 0]
        ax1.plot(results_df['k'], results_df['silhouette'], 'bo-', linewidth=2)
        ax1.set_xlabel('k')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Silhouette Score (Higher is Better)')
        ax1.grid(True, alpha=0.3)
        
        
        ax2 = axes[0, 1]
        ax2.plot(results_df['k'], results_df['calinski_harabasz'], 'go-', linewidth=2)
        ax2.set_xlabel('k')
        ax2.set_ylabel('CH Index')
        ax2.set_title('Calinski-Harabasz Index (Higher is Better)')
        ax2.grid(True, alpha=0.3)
        
        
        ax3 = axes[0, 2]
        ax3.plot(results_df['k'], results_df['davies_bouldin'], 'ro-', linewidth=2)
        ax3.set_xlabel('k')
        ax3.set_ylabel('DB Index')
        ax3.set_title('Davies-Bouldin Index (Lower is Better)')
        ax3.grid(True, alpha=0.3)
        
        
        ax4 = axes[1, 0]
        ax4.plot(results_df['k'], results_df['inertia'], 'mo-', linewidth=2)
        ax4.set_xlabel('k')
        ax4.set_ylabel('Inertia (WCSS)')
        ax4.set_title('Elbow Method')
        ax4.grid(True, alpha=0.3)
        
        
        ax5 = axes[1, 1]
        ax5.plot(results_df['k'], results_df['sil_norm'], 'b-', label='Silhouette', linewidth=2)
        ax5.plot(results_df['k'], results_df['ch_norm'], 'g-', label='CH Index', linewidth=2)
        ax5.plot(results_df['k'], results_df['db_norm'], 'r-', label='DB Index (inv)', linewidth=2)
        ax5.set_xlabel('k')
        ax5.set_ylabel('Normalized Score')
        ax5.set_title('Normalized Metrics Comparison')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        
        ax6 = axes[1, 2]
        best_k = results_df.loc[results_df['combined_score'].idxmax(), 'k']
        ax6.bar(results_df['k'], results_df['combined_score'], color='purple', alpha=0.7)
        ax6.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={int(best_k)}')
        ax6.set_xlabel('k')
        ax6.set_ylabel('Combined Score')
        ax6.set_title('Combined Score (Higher is Better)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nRecommended k based on combined metrics: {int(best_k)}")
        return results_df
