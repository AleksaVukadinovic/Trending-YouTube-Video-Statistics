# YouTube Trending Videos Clustering Project

A comprehensive data mining project that performs clustering analysis on the Trending YouTube Video Statistics dataset using multiple algorithms, dimensionality reduction techniques, and thorough evaluation.

## Project Description

This project analyzes trending YouTube videos to discover natural groupings based on engagement metrics, temporal patterns, and content characteristics. It implements 5 different clustering algorithms and compares their performance using standard evaluation metrics.

## Dataset

**Source:** [Kaggle - Trending YouTube Video Statistics](https://www.kaggle.com/datasets/datasnaek/youtube-new)

**Recommended file:** `DEvideos.csv` (German YouTube trending videos)

The dataset contains information about trending YouTube videos, including:
- Video metadata (title, channel, category)
- Engagement metrics (views, likes, dislikes, comments)
- Temporal information (publish time, trending date)
- Tags and descriptions

## Features

### Data Preprocessing
- Duplicate removal
- Missing value handling
- Date conversion
- Irrelevant column removal

### Feature Engineering (100+ features)
- **Temporal features:** day_of_week, month, trending_duration, publish_hour
- **Engagement metrics:** likes_per_view, comments_per_view, engagement_rate
- **Text features:** title_length, title_sentiment, tags_count
- **TF-IDF features:** 50 features from tags
- **Category encoding:** One-hot encoded categories
- **Statistical features:** numeric_std, numeric_mean, numeric_range

### Clustering Algorithms
1. **KMeans** - Partition-based clustering with elbow method for optimal k
2. **Agglomerative Clustering** - Hierarchical clustering with Ward linkage
3. **DBSCAN** - Density-based clustering with automatic eps selection
4. **Gaussian Mixture Models** - Probabilistic clustering
5. **Spectral Clustering** - Graph-based clustering

### Dimensionality Reduction
- **Full features** - All engineered features
- **PCA** - Principal Component Analysis (95% variance retained)
- **SelectKBest** - Top 50 features by F-score

### Evaluation Metrics
- Silhouette Score (higher is better)
- Davies-Bouldin Index (lower is better)
- Calinski-Harabasz Score (higher is better)

## Project Structure

```
youtube_clustering/
│
├── data/
│   ├── raw/              # Place raw CSV files here
│   └── processed/        # Processed data output
│
├── models/               # Saved clustering models
│   ├── full/
│   ├── pca/
│   └── selectkbest/
│
├── notebooks/            # Jupyter notebooks (optional)
│
├── results/              # Evaluation results and reports
│
├── visualizations/       # Generated plots
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data cleaning and loading
│   ├── feature_engineering.py  # Feature creation and scaling
│   ├── clustering.py           # Clustering algorithms
│   ├── evaluation.py           # Metrics and comparison
│   ├── visualization.py        # Plot generation
│   └── main.py                 # Main pipeline script
│
├── requirements.txt
└── README.md
```

## Installation

1. Clone or download this repository

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset from Kaggle and place the CSV file(s) in `data/raw/`

## How to Run

### Basic Usage

```bash
cd app/src
python main.py
```

The script will automatically download the dataset from Kaggle using `kagglehub` if not present in `data/raw/`.

### With Specific Data File

```bash
python main.py /path/to/your/data.csv
```

## Outputs

After running the pipeline, you will find:

### Data
- `data/processed/processed_data.csv` - Scaled feature matrix

### Models
- `models/full/` - Models trained on all features
- `models/pca/` - Models trained on PCA-reduced features
- `models/selectkbest/` - Models trained on SelectKBest features

Each folder contains:
- `{algorithm}_model.joblib` - Trained model
- `{algorithm}_labels.npy` - Cluster assignments

### Visualizations
- `clusters_2d.png` - 2D PCA projection of best model
- `clusters_3d.png` - 3D PCA projection of best model
- `cluster_distribution.png` - Bar chart of cluster sizes
- `correlation_heatmap.png` - Feature correlation matrix
- `elbow_curve.png` - KMeans elbow method plot
- `metrics_comparison.png` - Algorithm comparison chart
- Individual algorithm plots for each feature set

### Results
- `evaluation_results.csv` - Metrics for all model configurations
- `analysis_report.md` - Comprehensive text analysis

## Algorithm Descriptions

### KMeans
Partitions data into k clusters by minimizing within-cluster variance. The optimal k is determined using the elbow method.

### Agglomerative Clustering
Bottom-up hierarchical clustering that merges the closest clusters iteratively using Ward's minimum variance criterion.

### DBSCAN
Density-based algorithm that groups points in high-density regions and marks outliers as noise. Eps parameter is automatically tuned.

### Gaussian Mixture Models
Probabilistic model assuming data is generated from a mixture of Gaussian distributions. Provides soft cluster assignments.

### Spectral Clustering
Uses eigenvalues of the similarity matrix to reduce dimensionality before clustering. Effective for non-convex clusters.

## Reproducibility

All random operations use `RANDOM_SEED = 42` for reproducibility. To reproduce results:

1. Use the same dataset version
2. Run `main.py` without modifications
3. Results should be identical across runs

## Requirements

- Python 3.11+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- plotly >= 5.15.0
- scipy >= 1.11.0
- joblib >= 1.3.0
- textblob >= 0.17.1

