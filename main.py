import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from data_loader import YouTubeDataLoader
from preprocessor import DataPreprocessor
from visualizer import Visualizer
from cluster_engine import ClusterEngine
from cluster_evaluation import ClusterEvaluator
from feature_engineering import FeatureEngineer
from dimensionality_reduction import DimensionalityReducer
from feature_importance import FeatureImportanceAnalyzer
from anomaly_detection import AnomalyDetector
from regional_analysis import RegionalClusterAnalyzer
from predictive_models import ClusterPredictor, ViewCountPredictor
from cluster_profiler import ClusterProfiler

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def run_basic_analysis(clean_df):
    """Run basic EDA and clustering."""
    print("\n" + "="*60)
    print("BASIC EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    viz = Visualizer(clean_df)
    viz.plot_correlation_heatmap()
    viz.plot_category_popularity()
    viz.plot_engagement_by_region()
    
    print("\n" + "="*60)
    print("BASIC K-MEANS CLUSTERING")
    print("="*60)
    
    cluster_engine = ClusterEngine(clean_df)
    cluster_engine.prepare_features()
    cluster_engine.find_optimal_k(max_k=8)
    result_df = cluster_engine.apply_clustering(k=4)
    cluster_engine.visualize_clusters_pca()
    
    stats = cluster_engine.get_cluster_stats()
    pd.options.display.float_format = '{:,.2f}'.format
    print("\nCluster Statistics:")
    print(stats.T)
    
    return result_df, cluster_engine

def run_cluster_evaluation(X_scaled, feature_names):
    """Run comprehensive cluster evaluation."""
    print("\n" + "="*60)
    print("CLUSTER EVALUATION & STABILITY ANALYSIS")
    print("="*60)
    
    evaluator = ClusterEvaluator(X_scaled, feature_names)
    
    
    print("\nRunning Silhouette Analysis...")
    evaluator.plot_silhouette_analysis(k_range=range(2, 9))
    
    
    print("\nRunning Cluster Stability Analysis...")
    stability_results = evaluator.plot_stability_analysis(k=4, n_bootstrap=50)
    
    
    print("\nComprehensive K Selection Analysis...")
    k_results = evaluator.plot_optimal_k_analysis(k_range=range(2, 9))
    
    return evaluator, stability_results


def run_feature_engineering(clean_df):
    """Run advanced feature engineering."""
    print("\n" + "="*60)
    print("ADVANCED FEATURE ENGINEERING")
    print("="*60)
    
    fe = FeatureEngineer(clean_df)
    
    
    fe.add_temporal_features()
    fe.add_tfidf_features(text_column='title', n_components=10)
    fe.add_sentiment_features(text_column='title')
    fe.add_text_statistics(text_column='title')
    
    
    new_features = fe.get_all_new_features()
    print("\nEngineered Features Summary:")
    for category, features in new_features.items():
        print(f"  {category}: {len(features)} features")
    
    return fe.df, fe


def run_dimensionality_reduction(X_scaled, labels):
    """Run dimensionality reduction visualizations."""
    print("\n" + "="*60)
    print("DIMENSIONALITY REDUCTION & VISUALIZATION")
    print("="*60)
    
    reducer = DimensionalityReducer(X_scaled, labels)
    
    
    print("\nFitting PCA...")
    reducer.fit_pca(n_components=2)
    reducer.plot_pca_variance()
    
    
    print("\nFitting t-SNE...")
    reducer.fit_tsne(perplexity=30)
    
    
    print("\nFitting UMAP...")
    reducer.fit_umap(n_neighbors=15, min_dist=0.1)
    
    
    print("\nComparing Dimensionality Reduction Methods...")
    reducer.plot_all_methods()
    
    
    print("\nGenerating 3D Visualization...")
    reducer.plot_3d_visualization(method='pca')
    
    return reducer


def run_feature_importance(X_scaled, labels, feature_names):
    """Analyze feature importance for clustering."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    analyzer = FeatureImportanceAnalyzer(X_scaled, labels, feature_names)
    
    
    combined_ranking = analyzer.plot_feature_importance(top_n=12)
    print("\nTop Features (Combined Ranking):")
    print(combined_ranking.head(10))
    
    
    print("\nPlotting Feature Distributions by Cluster...")
    analyzer.plot_feature_distributions_by_cluster(top_n=6)
    
    
    print("\nAnalyzing Feature-Cluster Correlations...")
    analyzer.plot_feature_correlation_with_clusters()
    
    return analyzer


def run_anomaly_detection(X_scaled, clean_df, labels, feature_names):
    """Run anomaly detection analysis."""
    print("\n" + "="*60)
    print("ANOMALY DETECTION")
    print("="*60)
    
    detector = AnomalyDetector(X_scaled, clean_df, feature_names)
    
    
    detector.run_all_methods(contamination=0.05)
    
    
    ensemble_anomalies, votes = detector.ensemble_anomalies(min_votes=2)
    
    
    print("\nComparing Anomaly Detection Methods...")
    detector.plot_anomaly_comparison()
    
    
    print("\nVisualizing Anomalies in 2D...")
    detector.plot_anomalies_2d(labels=labels, method='isolation_forest')
    
    
    print("\nAnalyzing Anomaly Feature Patterns...")
    detector.plot_anomaly_feature_analysis(method='isolation_forest')
    
    
    print("\nTop Anomalous Videos:")
    top_anomalies = detector.get_anomaly_details(method='isolation_forest', top_n=10)
    if 'title' in top_anomalies.columns:
        print(top_anomalies[['title', 'views', 'anomaly_score']].head(10))
    
    return detector


def run_regional_analysis(clean_df, feature_cols):
    """Run cross-regional clustering analysis."""
    print("\n" + "="*60)
    print("REGIONAL CLUSTER ANALYSIS")
    print("="*60)
    
    regional = RegionalClusterAnalyzer(clean_df, feature_cols)
    
    
    regional.cluster_by_region(k=4)
    
    
    print("\nComparing Regional Clustering...")
    comparison = regional.plot_regional_comparison()
    
    
    print("\nGenerating Regional Cluster Profiles...")
    regional.plot_regional_cluster_profiles(top_regions=4)
    
    
    print("\nFinding Universal Patterns Across Regions...")
    pattern_mapping, universal_centroids = regional.find_universal_patterns(k=4)
    
    return regional


def run_predictive_models(X_scaled, labels, views, feature_names):
    """Train predictive models."""
    print("\n" + "="*60)
    print("PREDICTIVE MODELING")
    print("="*60)
    
    
    print("\n--- Cluster Prediction Model ---")
    cluster_predictor = ClusterPredictor(X_scaled, labels, feature_names)
    cluster_predictor.prepare_data()
    cluster_predictor.train_all_models()
    cluster_predictor.cross_validate('random_forest', cv=5)
    cluster_predictor.plot_model_comparison()
    
    
    print("\n--- View Count Prediction Model ---")
    view_predictor = ViewCountPredictor(X_scaled, views, feature_names, cluster_labels=labels)
    view_predictor.prepare_data(log_transform_target=True)
    view_predictor.train_all_models()
    view_predictor.plot_model_comparison()
    view_predictor.plot_feature_importance(top_n=12)
    
    return cluster_predictor, view_predictor


def run_cluster_profiling(clean_df, labels, feature_cols):
    """Generate cluster profiles and dashboard."""
    print("\n" + "="*60)
    print("CLUSTER PROFILING & BUSINESS INSIGHTS")
    print("="*60)
    
    profiler = ClusterProfiler(clean_df, labels, feature_cols)
    
    
    profiles = profiler.generate_cluster_profiles()
    print("\nCluster Profiles:")
    for cluster, profile in profiles.items():
        print(f"\nCluster {cluster}: {profile['description']}")
        print(f"  Size: {profile['size']:,} ({profile['percentage']:.1f}%)")
    
    
    print("\nGenerating Cluster Dashboard...")
    profiler.plot_cluster_dashboard()
    
    
    if 'trending_date' in clean_df.columns:
        print("\nAnalyzing Cluster Evolution Over Time...")
        profiler.plot_cluster_evolution()
    
    
    print("\nFinding Representative Videos per Cluster...")
    representatives = profiler.get_representative_videos(n_per_cluster=3)
    for cluster, rep_df in representatives.items():
        print(f"\nCluster {cluster} Representatives:")
        if 'title' in rep_df.columns:
            for _, row in rep_df.iterrows():
                print(f"  - {row['title'][:60]}... ({row['views']:,} views)")
    
    return profiler


def main():
    """Main execution pipeline."""
    print("="*60)
    print("YOUTUBE TRENDING VIDEO ANALYSIS - ML PIPELINE")
    print("="*60)
    
    
    print("\n[1/10] Downloading Dataset...")
    path = kagglehub.dataset_download("datasnaek/youtube-new")
    print(f"Path to dataset files: {path}")
    
    
    print("\n[2/10] Loading Data...")
    loader = YouTubeDataLoader(path)
    loader.load_categories()
    raw_df = loader.load_data()
    
    
    print("\n[3/10] Preprocessing Data...")
    preprocessor = DataPreprocessor(raw_df, loader.category_map)
    clean_df = preprocessor.clean_and_engineer()
    
    
    feature_cols = ['views', 'likes', 'dislikes', 'comment_count', 'days_to_trend', 'title_length']
    
    
    print("\n[4/10] Running Basic Analysis...")
    result_df, basic_cluster = run_basic_analysis(clean_df)
    
    
    X_scaled = basic_cluster.X_scaled
    labels = result_df['Cluster'].values
    model_features = ['log_views', 'log_likes', 'log_dislikes', 'log_comment_count', 'days_to_trend', 'title_length']
    
    
    print("\n[6/10] Evaluating Clusters...")
    evaluator, stability = run_cluster_evaluation(X_scaled, model_features)
    
    
    print("\n[7/10] Engineering Advanced Features...")
    enhanced_df, fe = run_feature_engineering(clean_df.copy())
    
    
    print("\n[8/10] Dimensionality Reduction & Visualization...")
    reducer = run_dimensionality_reduction(X_scaled, labels)
    
    
    print("\n[9/10] Analyzing Feature Importance...")
    importance_analyzer = run_feature_importance(X_scaled, labels, model_features)
    
    
    print("\n[10/10] Detecting Anomalies...")
    detector = run_anomaly_detection(X_scaled, result_df, labels, model_features)
    
    
    print("\n" + "="*60)
    print("ADDITIONAL ANALYSES")
    print("="*60)
    
    
    print("\nRunning Regional Analysis...")
    regional = run_regional_analysis(clean_df, feature_cols)
    
    
    print("\nTraining Predictive Models...")
    views = result_df['views'].values
    cluster_predictor, view_predictor = run_predictive_models(X_scaled, labels, views, model_features)
    
    
    print("\nGenerating Cluster Profiles...")
    profiler = run_cluster_profiling(result_df, labels, feature_cols)
    
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"""
    Summary:
    --------
    • Total videos analyzed: {len(clean_df):,}
    • Unique regions: {clean_df['region'].nunique()}
    • Clusters identified: {len(np.unique(labels))}
    • Best clustering algorithm: See comparison above
    • Anomalies detected: {detector.anomaly_results['ensemble']['n_anomalies']:,}
    • View prediction R²: {view_predictor.models['random_forest']['test_r2']:.3f}
    • Cluster prediction accuracy: {cluster_predictor.models['random_forest']['test_accuracy']:.3f}
    """)


if __name__ == "__main__":
    main()
