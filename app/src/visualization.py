"""
Visualization module for clustering results.
Creates 2D/3D scatter plots, distribution plots, heatmaps, and elbow curves.
Uses descriptive cluster names instead of numeric labels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Dict, List

from cluster_naming import generate_all_cluster_names, get_cluster_descriptions

plt.style.use('seaborn-v0_8-whitegrid')
RANDOM_SEED = 42


def plot_2d_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str] = None,
    title: str = "Cluster Visualization (2D PCA)",
    output_path: str = None,
    cluster_names: Dict[int, str] = None
) -> None:
    """Create 2D scatter plot of clusters using PCA projection with descriptive names."""
    if data.shape[1] > 2:
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        data_2d = pca.fit_transform(data)
        explained_var = sum(pca.explained_variance_ratio_)
        title += f" ({explained_var:.1%} variance)"
    else:
        data_2d = data
    
    if cluster_names is None and feature_names is not None:
        cluster_names = generate_all_cluster_names(data, labels, feature_names)
    elif cluster_names is None:
        cluster_names = {l: f'Cluster {l}' for l in set(labels)}
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 10)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = cluster_names.get(label, f'Cluster {label}')
        count = mask.sum()
        pct = count / len(labels) * 100
        legend_label = f'{name} ({count:,}, {pct:.1f}%)'
        
        if label == -1:
            ax.scatter(data_2d[mask, 0], data_2d[mask, 1],
                      c='gray', marker='x', s=30, alpha=0.5, label=legend_label)
        else:
            ax.scatter(data_2d[mask, 0], data_2d[mask, 1],
                      c=[colors[i % 10]], s=50, alpha=0.6, label=legend_label)
    
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved 2D plot to: {output_path}")
    
    plt.close()


def plot_3d_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str] = None,
    title: str = "Cluster Visualization (3D PCA)",
    output_path: str = None,
    cluster_names: Dict[int, str] = None
) -> None:
    """Create 3D scatter plot of clusters using PCA projection with descriptive names."""
    if data.shape[1] > 3:
        pca = PCA(n_components=3, random_state=RANDOM_SEED)
        data_3d = pca.fit_transform(data)
        explained_var = sum(pca.explained_variance_ratio_)
        title += f" ({explained_var:.1%} variance)"
    else:
        data_3d = data[:, :3] if data.shape[1] >= 3 else np.hstack([data, np.zeros((len(data), 3-data.shape[1]))])
    
    if cluster_names is None and feature_names is not None:
        cluster_names = generate_all_cluster_names(data, labels, feature_names)
    elif cluster_names is None:
        cluster_names = {l: f'Cluster {l}' for l in set(labels)}
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 10)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = cluster_names.get(label, f'Cluster {label}')
        count = mask.sum()
        pct = count / len(labels) * 100
        legend_label = f'{name} ({count:,}, {pct:.1f}%)'
        
        if label == -1:
            ax.scatter(data_3d[mask, 0], data_3d[mask, 1], data_3d[mask, 2],
                      c='gray', marker='x', s=20, alpha=0.5, label=legend_label)
        else:
            ax.scatter(data_3d[mask, 0], data_3d[mask, 1], data_3d[mask, 2],
                      c=[colors[i % 10]], s=40, alpha=0.6, label=legend_label)
    
    ax.set_xlabel('PC1', fontsize=10)
    ax.set_ylabel('PC2', fontsize=10)
    ax.set_zlabel('PC3', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D plot to: {output_path}")
    
    plt.close()


def plot_cluster_distribution(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str] = None,
    title: str = "Cluster Distribution",
    output_path: str = None,
    cluster_names: Dict[int, str] = None
) -> None:
    """Create bar plot showing cluster size distribution with descriptive names."""
    if cluster_names is None and feature_names is not None:
        cluster_names = generate_all_cluster_names(data, labels, feature_names)
    elif cluster_names is None:
        cluster_names = {l: f'Cluster {l}' for l in set(labels)}
    
    unique, counts = np.unique(labels, return_counts=True)
    
    sorted_indices = np.argsort(counts)[::-1]
    unique = unique[sorted_indices]
    counts = counts[sorted_indices]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['gray' if u == -1 else plt.cm.tab10(i % 10) 
              for i, u in enumerate(unique)]
    labels_str = [cluster_names.get(u, f'Cluster {u}') for u in unique]
    
    bars = ax.bar(range(len(labels_str)), counts, color=colors, edgecolor='black', alpha=0.8)
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        pct = count / len(labels) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
               f'{count:,}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(range(len(labels_str)))
    ax.set_xticklabels(labels_str, rotation=30, ha='right', fontsize=10)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Number of Videos', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved distribution plot to: {output_path}")
    
    plt.close()


def plot_cluster_characteristics(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    title: str = "Cluster Characteristics Heatmap",
    output_path: str = None,
    cluster_names: Dict[int, str] = None,
    top_n_features: int = 15
) -> None:
    """Create heatmap showing key feature differences between clusters."""
    if cluster_names is None:
        cluster_names = generate_all_cluster_names(data, labels, feature_names)
    
    unique_labels = sorted(set(labels) - {-1})
    
    global_means = np.mean(data, axis=0)
    global_stds = np.std(data, axis=0)
    global_stds[global_stds == 0] = 1
    
    z_scores_matrix = []
    for label in unique_labels:
        mask = labels == label
        cluster_means = np.mean(data[mask], axis=0)
        z_scores = (cluster_means - global_means) / global_stds
        z_scores_matrix.append(z_scores)
    
    z_scores_matrix = np.array(z_scores_matrix)
    
    feature_variance = np.var(z_scores_matrix, axis=0)
    top_feature_indices = np.argsort(feature_variance)[-top_n_features:]
    
    z_scores_subset = z_scores_matrix[:, top_feature_indices]
    feature_subset = [feature_names[i] for i in top_feature_indices]
    
    feature_labels = []
    for f in feature_subset:
        readable = f.replace('_', ' ').replace('tfidf tag', 'Tag').title()
        if len(readable) > 25:
            readable = readable[:22] + '...'
        feature_labels.append(readable)
    
    cluster_labels = [cluster_names.get(l, f'Cluster {l}') for l in unique_labels]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(
        z_scores_subset,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        xticklabels=feature_labels,
        yticklabels=cluster_labels,
        ax=ax,
        cbar_kws={'label': 'Z-Score (std from mean)'}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Clusters', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved characteristics heatmap to: {output_path}")
    
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
    
    readable_names = [f.replace('_', ' ').title()[:20] for f in feature_names]
    
    df = pd.DataFrame(data, columns=readable_names)
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
    
    ax.set_title(title, fontsize=14, fontweight='bold')
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
    ax.set_title(title, fontsize=14, fontweight='bold')
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
        ax.set_title(title, fontsize=11, fontweight='bold')
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
) -> Dict[str, Dict[int, str]]:
    """Generate all required visualizations with descriptive cluster names."""
    print("\n=== Creating Visualizations ===")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    best_algo = 'kmeans'
    best_feature_set = 'full'
    if 'avg_rank' in evaluation_df.columns:
        best_row = evaluation_df.loc[evaluation_df['avg_rank'].idxmin()]
        best_algo = best_row['algorithm']
        best_feature_set = best_row['feature_set']
    
    labels = results[best_feature_set][best_algo]['labels']
    
    cluster_names = generate_all_cluster_names(data, labels, feature_names)
    print(f"\nGenerated cluster names for {best_algo.upper()}:")
    for cid, name in cluster_names.items():
        if cid != -1:
            count = (labels == cid).sum()
            print(f"  - {name}: {count:,} videos")
    
    all_cluster_names = {f"{best_feature_set}_{best_algo}": cluster_names}
    
    plot_2d_clusters(
        data, labels, feature_names,
        title=f"2D Cluster Visualization ({best_algo.upper()} - {best_feature_set})",
        output_path=f"{output_dir}/clusters_2d.png",
        cluster_names=cluster_names
    )
    
    plot_3d_clusters(
        data, labels, feature_names,
        title=f"3D Cluster Visualization ({best_algo.upper()} - {best_feature_set})",
        output_path=f"{output_dir}/clusters_3d.png",
        cluster_names=cluster_names
    )
    
    plot_cluster_distribution(
        data, labels, feature_names,
        title=f"Cluster Distribution ({best_algo.upper()})",
        output_path=f"{output_dir}/cluster_distribution.png",
        cluster_names=cluster_names
    )
    
    plot_cluster_characteristics(
        data, labels, feature_names,
        title=f"Cluster Characteristics ({best_algo.upper()})",
        output_path=f"{output_dir}/cluster_characteristics.png",
        cluster_names=cluster_names
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
                    algo_cluster_names = generate_all_cluster_names(data, algo_labels, feature_names)
                    all_cluster_names[f"{feature_set}_{algo}"] = algo_cluster_names
                    
                    plot_2d_clusters(
                        data, algo_labels, feature_names,
                        title=f"{algo.upper()} ({feature_set})",
                        output_path=f"{output_dir}/{feature_set}_{algo}_2d.png",
                        cluster_names=algo_cluster_names
                    )
    
    print(f"\nAll visualizations saved to: {output_dir}")
    
    return all_cluster_names
