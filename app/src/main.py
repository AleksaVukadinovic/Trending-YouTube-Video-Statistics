import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

from data_preprocessing import load_raw_data, preprocess_data, save_processed_data
from feature_engineering import engineer_features
from clustering import run_clustering_pipeline, apply_pca
from evaluation import (
    evaluate_all_results, get_best_model, generate_cluster_summary,
    save_evaluation_results
)
from visualization import create_all_visualizations
from download_dataset import ensure_dataset
from cluster_naming import generate_all_cluster_names, get_cluster_descriptions

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def setup_paths(base_dir: str = None) -> dict:
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    else:
        base_dir = Path(base_dir)
    
    paths = {
        'base': base_dir,
        'raw_data': base_dir / 'data' / 'raw',
        'processed_data': base_dir / 'data' / 'processed',
        'models': base_dir / 'models',
        'visualizations': base_dir / 'visualizations',
        'results': base_dir / 'results'
    }
    
    for key, path in paths.items():
        if key != 'base':
            path.mkdir(parents=True, exist_ok=True)
    
    return paths


def find_data_file(raw_data_dir: Path) -> str:
    csv_files = list(raw_data_dir.glob('*.csv'))
    
    if csv_files:
        preferred = ['DEvideos.csv', 'USvideos.csv', 'GBvideos.csv']
        for pref in preferred:
            for csv in csv_files:
                if csv.name == pref:
                    return str(csv)
        return str(csv_files[0])
    
    return ensure_dataset()


def generate_analysis_report(
    evaluation_df: pd.DataFrame,
    results: dict,
    data: np.ndarray,
    feature_names: list,
    output_path: str
) -> str:
    report_lines = [
        "=" * 60,
        "YOUTUBE TRENDING VIDEOS CLUSTERING ANALYSIS REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "1. DATASET OVERVIEW",
        "-" * 40,
        f"   - Total samples: {data.shape[0]:,}",
        f"   - Total features: {data.shape[1]}",
        f"   - Feature types: Numeric (scaled)",
        "",
        "2. ALGORITHMS EVALUATED",
        "-" * 40,
        "   - KMeans",
        "   - Agglomerative Clustering",
        "   - DBSCAN",
        "   - Gaussian Mixture Models (GMM)",
        "   - Spectral Clustering",
        "",
        "3. FEATURE SET VARIANTS",
        "-" * 40,
        "   - Full features (all engineered features)",
        "   - PCA reduced (95% variance retained)",
        "   - SelectKBest reduced (top 50 features)",
        "",
        "4. EVALUATION METRICS",
        "-" * 40,
    ]
    
    for feature_set in ['full', 'pca', 'selectkbest']:
        subset = evaluation_df[evaluation_df['feature_set'] == feature_set]
        if len(subset) > 0:
            report_lines.append(f"\n   {feature_set.upper()} FEATURES:")
            for _, row in subset.iterrows():
                report_lines.append(
                    f"   {row['algorithm']:15} | "
                    f"Silhouette: {row['silhouette']:7.4f} | "
                    f"DB: {row['davies_bouldin']:7.4f} | "
                    f"CH: {row['calinski_harabasz']:10.2f}"
                )
    
    best = get_best_model(evaluation_df)
    report_lines.extend([
        "",
        "5. BEST PERFORMING MODEL",
        "-" * 40,
        f"   Algorithm: {best['algorithm'].upper()}",
        f"   Feature Set: {best['feature_set']}",
        f"   Silhouette Score: {best['silhouette']:.4f}",
        f"   Davies-Bouldin Index: {best['davies_bouldin']:.4f}",
        f"   Calinski-Harabasz Score: {best['calinski_harabasz']:.2f}",
        "",
        "6. CLUSTER INTERPRETATION",
        "-" * 40,
    ])
    
    best_labels = results[best['feature_set']][best['algorithm']]['labels']
    unique_labels = sorted(set(best_labels) - {-1})
    
    cluster_names = generate_all_cluster_names(data, best_labels, feature_names)
    cluster_descriptions = get_cluster_descriptions(data, best_labels, feature_names)
    
    global_means = np.mean(data, axis=0)
    global_stds = np.std(data, axis=0)
    global_stds[global_stds == 0] = 1
    
    for cluster_id in unique_labels:
        mask = best_labels == cluster_id
        cluster_size = mask.sum()
        cluster_pct = cluster_size / len(best_labels) * 100
        cluster_name = cluster_names.get(cluster_id, f'Cluster {cluster_id}')
        
        cluster_data = data[mask]
        cluster_means = cluster_data.mean(axis=0)
        z_scores = (cluster_means - global_means) / global_stds
        top_feature_indices = np.argsort(z_scores)[-5:][::-1]
        
        report_lines.append(f"\n   {cluster_name}:")
        report_lines.append(f"   - Size: {cluster_size:,} videos ({cluster_pct:.1f}%)")
        report_lines.append("   - Key characteristics (z-score from mean):")
        
        for idx in top_feature_indices:
            if idx < len(feature_names) and z_scores[idx] > 0.3:
                feat_name = feature_names[idx].replace('_', ' ').title()
                report_lines.append(
                    f"     * {feat_name}: {z_scores[idx]:+.2f}σ"
                )
    
    if -1 in best_labels:
        noise_count = list(best_labels).count(-1)
        report_lines.append(f"\n   Noise/Outliers: {noise_count:,} ({noise_count/len(best_labels)*100:.1f}%)")
    
    report_lines.extend([
        "",
        "7. DIMENSIONALITY REDUCTION IMPACT",
        "-" * 40,
    ])
    
    for algo in ['kmeans', 'agglomerative', 'gmm']:
        algo_results = evaluation_df[evaluation_df['algorithm'] == algo]
        if len(algo_results) >= 2:
            full_score = algo_results[algo_results['feature_set'] == 'full']['silhouette'].values
            pca_score = algo_results[algo_results['feature_set'] == 'pca']['silhouette'].values
            
            if len(full_score) > 0 and len(pca_score) > 0:
                diff = pca_score[0] - full_score[0]
                direction = "improved" if diff > 0 else "decreased"
                report_lines.append(
                    f"   {algo.upper()}: PCA {direction} silhouette by {abs(diff):.4f}"
                )
    
    report_lines.extend([
        "",
        "8. RECOMMENDATIONS",
        "-" * 40,
        f"   - Use {best['algorithm'].upper()} with {best['feature_set']} features for best results",
        "   - Consider the silhouette score for cluster quality assessment",
        "   - Review cluster distributions for balanced groupings",
        "",
        "=" * 60,
        "END OF REPORT",
        "=" * 60
    ])
    
    report = "\n".join(report_lines)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Analysis report saved to: {output_path}")
    return report


def main(data_file: str = None):
    print("=" * 60)
    print("YOUTUBE TRENDING VIDEOS CLUSTERING PROJECT")
    print("=" * 60)
    
    paths = setup_paths()
    
    if data_file is None:
        data_file = find_data_file(paths['raw_data'])
    
    print(f"\n[1/7] Loading raw data from: {data_file}")
    df_raw = load_raw_data(data_file)
    
    print(f"\n[2/7] Preprocessing data...")
    df_clean = preprocess_data(df_raw)
    
    print(f"\n[3/7] Engineering features...")
    df_scaled, scaler, tfidf, feature_names = engineer_features(df_clean)
    
    processed_path = paths['processed_data'] / 'processed_data.csv'
    df_scaled.to_csv(processed_path, index=False)
    print(f"Saved processed data to: {processed_path}")
    
    data = df_scaled.values
    
    print(f"\n[4/7] Running clustering pipeline...")
    results = run_clustering_pipeline(
        data,
        output_dir=str(paths['models']),
        n_clusters=None
    )
    
    data_variants = {}
    if 'pca' in results and 'reduced_data' in results['pca']:
        data_variants['pca'] = results['pca']['reduced_data']
    if 'selectkbest' in results and 'reduced_data' in results['selectkbest']:
        data_variants['selectkbest'] = results['selectkbest']['reduced_data']
    
    print(f"\n[5/7] Evaluating clustering results...")
    evaluation_df = evaluate_all_results(data, results, data_variants)
    
    eval_path = paths['results'] / 'evaluation_results.csv'
    save_evaluation_results(evaluation_df, str(eval_path))
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(evaluation_df.to_string(index=False))
    
    print(f"\n[6/7] Creating visualizations...")
    create_all_visualizations(
        data, results, evaluation_df, feature_names,
        output_dir=str(paths['visualizations'])
    )
    
    print(f"\n[7/7] Generating analysis report...")
    report_path = paths['results'] / 'analysis_report.md'
    report = generate_analysis_report(
        evaluation_df, results, data, feature_names, str(report_path)
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    best = get_best_model(evaluation_df)
    print(f"\nBest Model: {best['algorithm'].upper()} ({best['feature_set']})")
    print(f"Silhouette Score: {best['silhouette']:.4f}")
    
    print(f"\nOutputs saved to:")
    print(f"  - Processed data: {paths['processed_data']}")
    print(f"  - Models: {paths['models']}")
    print(f"  - Visualizations: {paths['visualizations']}")
    print(f"  - Results: {paths['results']}")
    
    return results, evaluation_df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
