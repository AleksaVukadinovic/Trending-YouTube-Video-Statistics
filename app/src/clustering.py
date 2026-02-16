"""
Clustering module for YouTube trending videos.
Implements 5 clustering algorithms with dimensionality reduction options.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def find_optimal_k_elbow(data: np.ndarray, k_range: range = range(2, 11)) -> tuple:
    """Find optimal k using elbow method. Returns optimal k and inertias."""
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    diffs = np.diff(inertias)
    second_diffs = np.diff(diffs)
    optimal_k = k_range[np.argmax(second_diffs) + 2] if len(second_diffs) > 0 else 5
    
    print(f"Optimal k from elbow method: {optimal_k}")
    return optimal_k, list(k_range), inertias


def apply_pca(data: np.ndarray, n_components: int = None, variance_ratio: float = 0.95) -> tuple:
    """Apply PCA dimensionality reduction."""
    if n_components is None:
        pca = PCA(n_components=variance_ratio, random_state=RANDOM_SEED)
    else:
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    
    reduced_data = pca.fit_transform(data)
    explained_var = sum(pca.explained_variance_ratio_)
    
    print(f"PCA: {data.shape[1]} -> {reduced_data.shape[1]} features "
          f"({explained_var:.2%} variance explained)")
    return reduced_data, pca


def apply_selectkbest(data: np.ndarray, k: int = 50, pseudo_labels: np.ndarray = None) -> tuple:
    """Apply SelectKBest feature selection."""
    if pseudo_labels is None:
        kmeans = KMeans(n_clusters=5, random_state=RANDOM_SEED, n_init=10)
        pseudo_labels = kmeans.fit_predict(data)
    
    k = min(k, data.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    reduced_data = selector.fit_transform(data, pseudo_labels)
    
    print(f"SelectKBest: {data.shape[1]} -> {reduced_data.shape[1]} features")
    return reduced_data, selector


def train_kmeans(data: np.ndarray, n_clusters: int = 5) -> tuple:
    """Train KMeans clustering model."""
    model = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_SEED,
        n_init=10,
        max_iter=300
    )
    labels = model.fit_predict(data)
    print(f"KMeans: {n_clusters} clusters, inertia={model.inertia_:.2f}")
    return model, labels


def train_agglomerative(data: np.ndarray, n_clusters: int = 5) -> tuple:
    """Train Agglomerative clustering model."""
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    labels = model.fit_predict(data)
    print(f"Agglomerative: {n_clusters} clusters")
    return model, labels


def train_dbscan(data: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> tuple:
    """Train DBSCAN clustering model."""
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        n_jobs=-1
    )
    labels = model.fit_predict(data)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points (eps={eps})")
    return model, labels


def find_optimal_dbscan_eps(data: np.ndarray, sample_size: int = 1000) -> float:
    """Find optimal eps for DBSCAN using k-distance graph."""
    from sklearn.neighbors import NearestNeighbors
    
    if len(data) > sample_size:
        idx = np.random.choice(len(data), sample_size, replace=False)
        sample_data = data[idx]
    else:
        sample_data = data
    
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(sample_data)
    distances, _ = nn.kneighbors(sample_data)
    
    distances = np.sort(distances[:, -1])
    gradient = np.gradient(distances)
    elbow_idx = np.argmax(gradient)
    optimal_eps = distances[elbow_idx]
    
    optimal_eps = max(0.3, min(optimal_eps, 2.0))
    print(f"Optimal DBSCAN eps: {optimal_eps:.3f}")
    return optimal_eps


def train_gmm(data: np.ndarray, n_components: int = 5) -> tuple:
    """Train Gaussian Mixture Model."""
    model = GaussianMixture(
        n_components=n_components,
        random_state=RANDOM_SEED,
        covariance_type='full',
        max_iter=200
    )
    model.fit(data)
    labels = model.predict(data)
    print(f"GMM: {n_components} components, BIC={model.bic(data):.2f}")
    return model, labels


def train_spectral(data: np.ndarray, n_clusters: int = 5) -> tuple:
    """Train Spectral clustering model."""
    n_samples = min(5000, len(data))
    if len(data) > n_samples:
        idx = np.random.choice(len(data), n_samples, replace=False)
        sample_data = data[idx]
    else:
        sample_data = data
        idx = np.arange(len(data))
    
    model = SpectralClustering(
        n_clusters=n_clusters,
        random_state=RANDOM_SEED,
        affinity='nearest_neighbors',
        n_neighbors=10,
        assign_labels='kmeans'
    )
    sample_labels = model.fit_predict(sample_data)
    
    if len(data) > n_samples:
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(sample_data, sample_labels)
        labels = knn.predict(data)
    else:
        labels = sample_labels
    
    print(f"Spectral: {n_clusters} clusters")
    return model, labels


def save_model(model, labels: np.ndarray, model_name: str, output_dir: str) -> None:
    """Save model and labels to disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    model_path = Path(output_dir) / f"{model_name}_model.joblib"
    labels_path = Path(output_dir) / f"{model_name}_labels.npy"
    
    joblib.dump(model, model_path)
    np.save(labels_path, labels)
    print(f"Saved {model_name} to {output_dir}")


def train_all_models(data: np.ndarray, n_clusters: int = 5, output_dir: str = "models") -> dict:
    """Train all clustering models and return results."""
    print(f"\n=== Training Clustering Models (n_clusters={n_clusters}) ===")
    
    results = {}
    
    model, labels = train_kmeans(data, n_clusters)
    results['kmeans'] = {'model': model, 'labels': labels}
    save_model(model, labels, 'kmeans', output_dir)
    
    model, labels = train_agglomerative(data, n_clusters)
    results['agglomerative'] = {'model': model, 'labels': labels}
    save_model(model, labels, 'agglomerative', output_dir)
    
    eps = find_optimal_dbscan_eps(data)
    model, labels = train_dbscan(data, eps=eps, min_samples=5)
    results['dbscan'] = {'model': model, 'labels': labels}
    save_model(model, labels, 'dbscan', output_dir)
    
    model, labels = train_gmm(data, n_clusters)
    results['gmm'] = {'model': model, 'labels': labels}
    save_model(model, labels, 'gmm', output_dir)
    
    model, labels = train_spectral(data, n_clusters)
    results['spectral'] = {'model': model, 'labels': labels}
    save_model(model, labels, 'spectral', output_dir)
    
    return results


def run_clustering_pipeline(
    data: np.ndarray,
    output_dir: str = "models",
    n_clusters: int = None
) -> dict:
    """Run complete clustering pipeline with dimensionality reduction variants."""
    print("\n=== Running Clustering Pipeline ===")
    
    all_results = {}
    
    if n_clusters is None:
        n_clusters, k_range, inertias = find_optimal_k_elbow(data)
        all_results['elbow'] = {'k_range': k_range, 'inertias': inertias, 'optimal_k': n_clusters}
    
    print("\n--- Full Features ---")
    all_results['full'] = train_all_models(
        data, n_clusters, f"{output_dir}/full"
    )
    
    print("\n--- PCA Reduced ---")
    pca_data, pca_model = apply_pca(data, variance_ratio=0.95)
    all_results['pca'] = train_all_models(
        pca_data, n_clusters, f"{output_dir}/pca"
    )
    all_results['pca']['pca_model'] = pca_model
    all_results['pca']['reduced_data'] = pca_data
    
    print("\n--- SelectKBest Reduced ---")
    k_best = min(50, data.shape[1] // 2)
    kbest_data, selector = apply_selectkbest(data, k=k_best)
    all_results['selectkbest'] = train_all_models(
        kbest_data, n_clusters, f"{output_dir}/selectkbest"
    )
    all_results['selectkbest']['selector'] = selector
    all_results['selectkbest']['reduced_data'] = kbest_data
    
    all_results['n_clusters'] = n_clusters
    
    return all_results
