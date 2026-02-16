"""
Visualization module for clustering results.
Creates 2D/3D scatter plots, distribution plots, heatmaps, and elbow curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
RANDOM_SEED = 42


def plot_2d_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    title: str = "Cluster Visualization (2D PCA)",
    output_path: str = None
) -> None:
    """Create 2D scatter plot of clusters using PCA projection."""
    if data.shape[1] > 2:
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        data_2d = pca.fit_transform(data)
        explained_var = sum(pca.explained_variance_ratio_)
        title += f" ({explained_var:.1%} variance)"
    else:
        data_2d = data
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax.scatter(data_2d[mask, 0], data_2d[mask, 1],
                      c='gray', marker='x', s=30, alpha=0.5, label='Noise')
        else:
            ax.scatter(data_2d[mask, 0], data_2d[mask, 1],
                      c=[colors[i]], s=50, alpha=0.6, label=f'Cluster {label}')
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved 2D plot to: {output_path}")
    
    plt.close()


def plot_3d_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    title: str = "Cluster Visualization (3D PCA)",
    output_path: str = None
) -> None:
    """Create 3D scatter plot of clusters using PCA projection."""
    if data.shape[1] > 3:
        pca = PCA(n_components=3, random_state=RANDOM_SEED)
        data_3d = pca.fit_transform(data)
        explained_var = sum(pca.explained_variance_ratio_)
        title += f" ({explained_var:.1%} variance)"
    else:
        data_3d = data[:, :3] if data.shape[1] >= 3 else np.hstack([data, np.zeros((len(data), 3-data.shape[1]))])
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax.scatter(data_3d[mask, 0], data_3d[mask, 1], data_3d[mask, 2],
                      c='gray', marker='x', s=20, alpha=0.5, label='Noise')
        else:
            ax.scatter(data_3d[mask, 0], data_3d[mask, 1], data_3d[mask, 2],
                      c=[colors[i]], s=40, alpha=0.6, label=f'Cluster {label}')
    
    ax.set_xlabel('PC1', fontsize=10)
    ax.set_ylabel('PC2', fontsize=10)
    ax.set_zlabel('PC3', fontsize=10)
    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D plot to: {output_path}")
    
    plt.close()


def plot_cluster_distribution(
    labels: np.ndarray,
    title: str = "Cluster Distribution",
    output_path: str = None
) -> None:
    """Create bar plot showing cluster size distribution."""
    unique, counts = np.unique(labels, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['gray' if u == -1 else plt.cm.tab10(i % 10) 
              for i, u in enumerate(unique)]
    labels_str = ['Noise' if u == -1 else f'Cluster {u}' for u in unique]
    
    bars = ax.bar(labels_str, counts, color=colors, edgecolor='black', alpha=0.7)
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
               f'{count}\n({count/len(labels)*100:.1f}%)',
               ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved distribution plot to: {output_path}")
    
    plt.close()


def plot_correlation_heatmap(
    data: np.ndarray,
    feature_names: list = None,
    title: str = "Feature Correlation Heatmap",
    output_path: str = None,
    max_features: int = 30
) -> None:
    """Create correlation heatmap for features."""
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(data.shape[1])]
    
    if data.shape[1] > max_features:
        variances = np.var(data, axis=0)
        top_indices = np.argsort(variances)[-max_features:]
        data = data[:, top_indices]
        feature_names = [feature_names[i] for i in top_indices]
    
    df = pd.DataFrame(data, columns=feature_names)
    corr_matrix = df.corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=False,
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved correlation heatmap to: {output_path}")
    
    plt.close()


def plot_elbow_curve(
    k_range: list,
    inertias: list,
    optimal_k: int = None,
    title: str = "Elbow Method for Optimal k",
    output_path: str = None
) -> None:
    """Create elbow curve plot for KMeans."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_range, inertias, 'b-o', linewidth=2, markersize=8)
    
    if optimal_k and optimal_k in k_range:
        idx = k_range.index(optimal_k)
        ax.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, 
                   label=f'Optimal k = {optimal_k}')
        ax.scatter([optimal_k], [inertias[idx]], c='red', s=200, zorder=5)
    
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved elbow curve to: {output_path}")
    
    plt.close()


def plot_metrics_comparison(
    evaluation_df: pd.DataFrame,
    output_path: str = None
) -> None:
    """Create comparison plot of clustering metrics across algorithms."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
    titles = ['Silhouette Score (higher is better)', 
              'Davies-Bouldin Index (lower is better)',
              'Calinski-Harabasz Score (higher is better)']
    
    for ax, metric, title in zip(axes, metrics, titles):
        pivot_df = evaluation_df.pivot(
            index='algorithm', 
            columns='feature_set', 
            values=metric
        )
        pivot_df.plot(kind='bar', ax=ax, width=0.8, edgecolor='black')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Algorithm', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.legend(title='Feature Set', fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics comparison to: {output_path}")
    
    plt.close()


def create_all_visualizations(
    data: np.ndarray,
    results: dict,
    evaluation_df: pd.DataFrame,
    feature_names: list,
    output_dir: str = "visualizations"
) -> None:
    """Generate all required visualizations."""
    print("\n=== Creating Visualizations ===")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    best_algo = 'kmeans'
    best_feature_set = 'full'
    if 'avg_rank' in evaluation_df.columns:
        best_row = evaluation_df.loc[evaluation_df['avg_rank'].idxmin()]
        best_algo = best_row['algorithm']
        best_feature_set = best_row['feature_set']
    
    labels = results[best_feature_set][best_algo]['labels']
    
    plot_2d_clusters(
        data, labels,
        title=f"2D Cluster Visualization ({best_algo.upper()} - {best_feature_set})",
        output_path=f"{output_dir}/clusters_2d.png"
    )
    
    plot_3d_clusters(
        data, labels,
        title=f"3D Cluster Visualization ({best_algo.upper()} - {best_feature_set})",
        output_path=f"{output_dir}/clusters_3d.png"
    )
    
    plot_cluster_distribution(
        labels,
        title=f"Cluster Distribution ({best_algo.upper()})",
        output_path=f"{output_dir}/cluster_distribution.png"
    )
    
    plot_correlation_heatmap(
        data, feature_names,
        title="Feature Correlation Heatmap",
        output_path=f"{output_dir}/correlation_heatmap.png"
    )
    
    if 'elbow' in results:
        plot_elbow_curve(
            results['elbow']['k_range'],
            results['elbow']['inertias'],
            results['elbow']['optimal_k'],
            output_path=f"{output_dir}/elbow_curve.png"
        )
    
    plot_metrics_comparison(
        evaluation_df,
        output_path=f"{output_dir}/metrics_comparison.png"
    )
    
    for feature_set in ['full', 'pca', 'selectkbest']:
        if feature_set in results:
            for algo in ['kmeans', 'agglomerative', 'dbscan', 'gmm', 'spectral']:
                if algo in results[feature_set]:
                    algo_labels = results[feature_set][algo]['labels']
                    plot_2d_clusters(
                        data, algo_labels,
                        title=f"{algo.upper()} ({feature_set})",
                        output_path=f"{output_dir}/{feature_set}_{algo}_2d.png"
                    )
    
    print(f"All visualizations saved to: {output_dir}")
