import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')


try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Note: umap-learn not installed. Install with: pip install umap-learn")


class DimensionalityReducer:
    """
    Dimensionality reduction and visualization using PCA, t-SNE, and UMAP.
    """

    def __init__(self, X_scaled, labels=None):
        self.X_scaled = X_scaled
        self.labels = labels
        self.pca_result = None
        self.tsne_result = None
        self.umap_result = None

    def fit_pca(self, n_components=2):
        """Fit PCA for dimensionality reduction."""
        pca = PCA(n_components=n_components)
        self.pca_result = pca.fit_transform(self.X_scaled)
        self.pca_model = pca
        return self.pca_result

    def fit_tsne(self, n_components=2, perplexity=30, n_iter=1000, random_state=42):
        """Fit t-SNE for dimensionality reduction."""
        print(f"Fitting t-SNE (perplexity={perplexity})... This may take a while.")
        
        
        n_samples = len(self.X_scaled)
        if n_samples > 10000:
            print(f"Sampling 10000 points from {n_samples} for t-SNE...")
            sample_idx = np.random.choice(n_samples, 10000, replace=False)
            X_sample = self.X_scaled[sample_idx]
            labels_sample = self.labels[sample_idx] if self.labels is not None else None
        else:
            X_sample = self.X_scaled
            labels_sample = self.labels
            sample_idx = None
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state,
            init='pca',
            learning_rate='auto'
        )
        
        self.tsne_result = tsne.fit_transform(X_sample)
        self.tsne_labels = labels_sample
        self.tsne_sample_idx = sample_idx
        
        return self.tsne_result

    def fit_umap(self, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
        """Fit UMAP for dimensionality reduction."""
        if not UMAP_AVAILABLE:
            print("UMAP not available. Please install: pip install umap-learn")
            return None
        
        print(f"Fitting UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
        
        umap_model = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric='euclidean'
        )
        
        self.umap_result = umap_model.fit_transform(self.X_scaled)
        self.umap_model = umap_model
        
        return self.umap_result

    def plot_all_methods(self, figsize=(18, 5)):
        """Plot all dimensionality reduction methods side by side."""
        n_plots = 1 + (self.tsne_result is not None) + (self.umap_result is not None)
        
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        
        if self.pca_result is not None:
            ax = axes[plot_idx]
            scatter = ax.scatter(
                self.pca_result[:, 0], 
                self.pca_result[:, 1],
                c=self.labels, 
                cmap='viridis', 
                alpha=0.6, 
                s=10
            )
            ax.set_title('PCA Projection')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            plt.colorbar(scatter, ax=ax, label='Cluster')
            plot_idx += 1
        
        
        if self.tsne_result is not None:
            ax = axes[plot_idx]
            labels_to_plot = self.tsne_labels if self.tsne_labels is not None else self.labels
            scatter = ax.scatter(
                self.tsne_result[:, 0], 
                self.tsne_result[:, 1],
                c=labels_to_plot, 
                cmap='viridis', 
                alpha=0.6, 
                s=10
            )
            ax.set_title('t-SNE Projection')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=ax, label='Cluster')
            plot_idx += 1
        
        
        if self.umap_result is not None:
            ax = axes[plot_idx]
            scatter = ax.scatter(
                self.umap_result[:, 0], 
                self.umap_result[:, 1],
                c=self.labels, 
                cmap='viridis', 
                alpha=0.6, 
                s=10
            )
            ax.set_title('UMAP Projection')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        plt.tight_layout()
        plt.show()

    def plot_pca_variance(self, n_components=None):
        """Plot PCA explained variance."""
        n_components = n_components or min(20, self.X_scaled.shape[1])
        
        pca_full = PCA(n_components=n_components)
        pca_full.fit(self.X_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        
        ax1 = axes[0]
        ax1.bar(range(1, n_components + 1), pca_full.explained_variance_ratio_, alpha=0.7)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('PCA Explained Variance by Component')
        ax1.set_xticks(range(1, n_components + 1))
        
        
        ax2 = axes[1]
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        ax2.plot(range(1, n_components + 1), cumsum, 'bo-', linewidth=2)
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        ax2.axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.set_xticks(range(1, n_components + 1))
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        
        n_95 = np.argmax(cumsum >= 0.95) + 1
        print(f"Components needed for 95% variance: {n_95}")
        
        return pca_full.explained_variance_ratio_

    def plot_tsne_perplexity_comparison(self, perplexities=[5, 15, 30, 50]):
        """Compare t-SNE results with different perplexity values."""
        n_perp = len(perplexities)
        fig, axes = plt.subplots(1, n_perp, figsize=(5*n_perp, 5))
        
        
        n_samples = len(self.X_scaled)
        if n_samples > 5000:
            sample_idx = np.random.choice(n_samples, 5000, replace=False)
            X_sample = self.X_scaled[sample_idx]
            labels_sample = self.labels[sample_idx] if self.labels is not None else None
        else:
            X_sample = self.X_scaled
            labels_sample = self.labels
        
        for i, perp in enumerate(perplexities):
            print(f"Fitting t-SNE with perplexity={perp}...")
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
            result = tsne.fit_transform(X_sample)
            
            ax = axes[i]
            scatter = ax.scatter(result[:, 0], result[:, 1], c=labels_sample, 
                               cmap='viridis', alpha=0.6, s=10)
            ax.set_title(f't-SNE (perplexity={perp})')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
        
        plt.tight_layout()
        plt.show()

    def plot_umap_parameter_comparison(self, n_neighbors_list=[5, 15, 30, 50], min_dist_list=[0.0, 0.1, 0.25, 0.5]):
        """Compare UMAP results with different parameters."""
        if not UMAP_AVAILABLE:
            print("UMAP not available")
            return
        
        fig, axes = plt.subplots(len(n_neighbors_list), len(min_dist_list), 
                                figsize=(4*len(min_dist_list), 4*len(n_neighbors_list)))
        
        for i, n_neighbors in enumerate(n_neighbors_list):
            for j, min_dist in enumerate(min_dist_list):
                print(f"Fitting UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
                
                umap_model = UMAP(n_components=2, n_neighbors=n_neighbors, 
                                 min_dist=min_dist, random_state=42)
                result = umap_model.fit_transform(self.X_scaled)
                
                ax = axes[i, j]
                scatter = ax.scatter(result[:, 0], result[:, 1], c=self.labels,
                                   cmap='viridis', alpha=0.6, s=5)
                ax.set_title(f'nn={n_neighbors}, md={min_dist}', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle('UMAP Parameter Comparison', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_3d_visualization(self, method='pca'):
        """Create 3D visualization of clusters."""
        from mpl_toolkits.mplot3d import Axes3D
        
        if method == 'pca':
            pca_3d = PCA(n_components=3)
            result = pca_3d.fit_transform(self.X_scaled)
            title = 'PCA 3D Projection'
        elif method == 'tsne':
            print("Fitting 3D t-SNE...")
            tsne_3d = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto')
            
            if len(self.X_scaled) > 5000:
                idx = np.random.choice(len(self.X_scaled), 5000, replace=False)
                result = tsne_3d.fit_transform(self.X_scaled[idx])
                labels = self.labels[idx] if self.labels is not None else None
            else:
                result = tsne_3d.fit_transform(self.X_scaled)
                labels = self.labels
            title = 't-SNE 3D Projection'
        elif method == 'umap' and UMAP_AVAILABLE:
            print("Fitting 3D UMAP...")
            umap_3d = UMAP(n_components=3, random_state=42)
            result = umap_3d.fit_transform(self.X_scaled)
            labels = self.labels
            title = 'UMAP 3D Projection'
        else:
            print(f"Method {method} not available")
            return
        
        if method == 'pca':
            labels = self.labels
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            result[:, 0], result[:, 1], result[:, 2],
            c=labels, cmap='viridis', alpha=0.6, s=10
        )
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_title(title)
        
        plt.colorbar(scatter, label='Cluster', shrink=0.5)
        plt.tight_layout()
        plt.show()

    def interactive_3d_plot(self, method='pca'):
        """Create interactive 3D plot using plotly if available."""
        try:
            import plotly.express as px
            
            if method == 'pca':
                pca_3d = PCA(n_components=3)
                result = pca_3d.fit_transform(self.X_scaled)
            elif method == 'umap' and UMAP_AVAILABLE:
                umap_3d = UMAP(n_components=3, random_state=42)
                result = umap_3d.fit_transform(self.X_scaled)
            else:
                print(f"Method {method} not supported for interactive plot")
                return
            
            df = pd.DataFrame(result, columns=['Dim1', 'Dim2', 'Dim3'])
            df['Cluster'] = self.labels.astype(str) if self.labels is not None else '0'
            
            fig = px.scatter_3d(df, x='Dim1', y='Dim2', z='Dim3', color='Cluster',
                               title=f'{method.upper()} 3D Interactive Visualization',
                               opacity=0.6)
            fig.show()
            
        except ImportError:
            print("Plotly not available. Install with: pip install plotly")
            self.plot_3d_visualization(method)
