"""
Evaluation module for clustering results.
Implements clustering metrics and comparison analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path

RANDOM_SEED = 42


def calculate_silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    """Calculate silhouette score. Returns -1 if invalid."""
    unique_labels = set(labels)
    unique_labels.discard(-1)
    
    if len(unique_labels) < 2:
        return -1.0
    
    valid_mask = labels != -1
    if valid_mask.sum() < 2:
        return -1.0
    
    try:
        return silhouette_score(data[valid_mask], labels[valid_mask])
    except Exception:
        return -1.0


def calculate_davies_bouldin(data: np.ndarray, labels: np.ndarray) -> float:
    """Calculate Davies-Bouldin index. Lower is better. Returns -1 if invalid."""
    unique_labels = set(labels)
    unique_labels.discard(-1)
    
    if len(unique_labels) < 2:
        return -1.0
    
    valid_mask = labels != -1
    if valid_mask.sum() < 2:
        return -1.0
    
    try:
        return davies_bouldin_score(data[valid_mask], labels[valid_mask])
    except Exception:
        return -1.0


def calculate_calinski_harabasz(data: np.ndarray, labels: np.ndarray) -> float:
    """Calculate Calinski-Harabasz score. Higher is better. Returns -1 if invalid."""
    unique_labels = set(labels)
    unique_labels.discard(-1)
    
    if len(unique_labels) < 2:
        return -1.0
    
    valid_mask = labels != -1
    if valid_mask.sum() < 2:
        return -1.0
    
    try:
        return calinski_harabasz_score(data[valid_mask], labels[valid_mask])
    except Exception:
        return -1.0


def evaluate_clustering(data: np.ndarray, labels: np.ndarray) -> dict:
    """Calculate all clustering metrics."""
    return {
        'silhouette': calculate_silhouette(data, labels),
        'davies_bouldin': calculate_davies_bouldin(data, labels),
        'calinski_harabasz': calculate_calinski_harabasz(data, labels),
        'n_clusters': len(set(labels) - {-1}),
        'n_noise': list(labels).count(-1) if -1 in labels else 0
    }


def evaluate_all_results(data: np.ndarray, results: dict, data_variants: dict = None) -> pd.DataFrame:
    """Evaluate all clustering results and create comparison table."""
    print("\n=== Evaluating Clustering Results ===")
    
    evaluation_rows = []
    
    for feature_set in ['full', 'pca', 'selectkbest']:
        if feature_set not in results:
            continue
        
        if feature_set == 'full':
            eval_data = data
        elif feature_set in data_variants:
            eval_data = data_variants[feature_set]
        else:
            eval_data = results[feature_set].get('reduced_data', data)
        
        for algo_name in ['kmeans', 'agglomerative', 'dbscan', 'gmm', 'spectral']:
            if algo_name not in results[feature_set]:
                continue
            
            labels = results[feature_set][algo_name]['labels']
            metrics = evaluate_clustering(eval_data, labels)
            
            evaluation_rows.append({
                'algorithm': algo_name,
                'feature_set': feature_set,
                'silhouette': metrics['silhouette'],
                'davies_bouldin': metrics['davies_bouldin'],
                'calinski_harabasz': metrics['calinski_harabasz'],
                'n_clusters': metrics['n_clusters'],
                'n_noise': metrics['n_noise']
            })
    
    df = pd.DataFrame(evaluation_rows)
    
    df['silhouette_rank'] = df.groupby('feature_set')['silhouette'].rank(ascending=False)
    df['db_rank'] = df.groupby('feature_set')['davies_bouldin'].rank(ascending=True)
    df['ch_rank'] = df.groupby('feature_set')['calinski_harabasz'].rank(ascending=False)
    df['avg_rank'] = (df['silhouette_rank'] + df['db_rank'] + df['ch_rank']) / 3
    
    print(f"\nEvaluation complete. {len(df)} model configurations evaluated.")
    return df


def get_best_model(evaluation_df: pd.DataFrame) -> dict:
    """Identify the best performing model based on average rank."""
    best_row = evaluation_df.loc[evaluation_df['avg_rank'].idxmin()]
    return {
        'algorithm': best_row['algorithm'],
        'feature_set': best_row['feature_set'],
        'silhouette': best_row['silhouette'],
        'davies_bouldin': best_row['davies_bouldin'],
        'calinski_harabasz': best_row['calinski_harabasz'],
        'avg_rank': best_row['avg_rank']
    }


def get_cluster_statistics(data: np.ndarray, labels: np.ndarray, feature_names: list = None) -> pd.DataFrame:
    """Calculate statistics for each cluster."""
    df = pd.DataFrame(data, columns=feature_names)
    df['cluster'] = labels
    
    valid_df = df[df['cluster'] != -1]
    
    stats = valid_df.groupby('cluster').agg(['mean', 'std', 'count']).reset_index()
    
    return stats


def generate_cluster_summary(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: list = None
) -> str:
    """Generate a text summary of cluster characteristics."""
    unique_labels = sorted(set(labels) - {-1})
    
    summary_lines = ["Cluster Characteristics Summary", "=" * 40]
    
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_data = data[mask]
        
        summary_lines.append(f"\nCluster {cluster_id}:")
        summary_lines.append(f"  - Size: {mask.sum()} samples ({mask.sum()/len(labels)*100:.1f}%)")
        
        if feature_names and len(feature_names) <= data.shape[1]:
            means = cluster_data.mean(axis=0)
            top_indices = np.argsort(means)[-5:][::-1]
            
            summary_lines.append("  - Top features (by mean):")
            for idx in top_indices:
                if idx < len(feature_names):
                    summary_lines.append(f"    * {feature_names[idx]}: {means[idx]:.3f}")
    
    if -1 in labels:
        noise_count = list(labels).count(-1)
        summary_lines.append(f"\nNoise points: {noise_count} ({noise_count/len(labels)*100:.1f}%)")
    
    return "\n".join(summary_lines)


def save_evaluation_results(
    evaluation_df: pd.DataFrame,
    output_path: str
) -> None:
    """Save evaluation results to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    evaluation_df.to_csv(output_path, index=False)
    print(f"Saved evaluation results to: {output_path}")
