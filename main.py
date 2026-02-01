import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from data_loader import YouTubeDataLoader
from preprocessor import DataPreprocessor
from visualizer import Visualizer
from cluster_engine import ClusterEngine


warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def main():
    path = kagglehub.dataset_download("datasnaek/youtube-new")
    print("Path to dataset files:", path)

    loader = YouTubeDataLoader(path)
    loader.load_categories()
    raw_df = loader.load_data()

    preprocessor = DataPreprocessor(raw_df, loader.category_map)
    clean_df = preprocessor.clean_and_engineer()

    viz = Visualizer(clean_df)
    print("\n--- Generating Exploratory Visualizations ---")
    viz.plot_correlation_heatmap()
    viz.plot_category_popularity()
    viz.plot_engagement_by_region()

    print("\n--- Starting Clustering Analysis ---")
    cluster_engine = ClusterEngine(clean_df)
    cluster_engine.prepare_features()

    cluster_engine.find_optimal_k(max_k=8)
    result_df = cluster_engine.apply_clustering(k=4)
    cluster_engine.visualize_clusters_pca()

    print("\n--- Cluster Business Profiles ---")
    stats = cluster_engine.get_cluster_stats()

    pd.options.display.float_format = '{:,.2f}'.format
    print(stats.T)

    print("\n--- Analyst Interpretation Guide ---")
    print("Based on typical cluster centers, look for these groups in the table above:")
    print("1. 'Viral Hits': Extremely high views/likes, short time to trend.")
    print("2. 'Controversial/News': High view count but higher dislike ratios or comment activity.")
    print("3. 'Niche/Community': Lower views but very high engagement rates (likes/comments per view).")
    print("4. 'Standard Content': Moderate metrics across the board (the majority of videos).")

if __name__ == "__main__":
    main()
