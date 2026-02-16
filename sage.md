# Prompt for Coding Agent — Clustering Project (Trending YouTube Video Statistics)

You are a senior data scientist and Python developer. Your task is to implement a complete data mining project for a university course on the topic of clustering using Python 3.11.

## Objective

Perform clustering on the Trending YouTube Video Statistics dataset and compare results obtained using multiple algorithms with proper preprocessing, visualization, and analysis.

Dataset:
https://www.kaggle.com/datasets/datasnaek/youtube-new?select=DEvideos.csv

Use a CSV file (e.g., DEvideos.csv or a combination of files if needed so that the dataset has ≥100 attributes after feature engineering).

---

## General Requirements

The implementation must:

* Use Python 3.11
* Use standard data science libraries (pandas, numpy, scikit-learn, matplotlib, seaborn, plotly if needed)
* Be clearly structured and readable
* Avoid unnecessary complexity
* Be reproducible (set random seeds)
* Include clear comments
* Be modular but not overengineered
* Run from the command line

---

## Project Structure

Create the following structure:

```
youtube_clustering/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

## 1. Data Preprocessing (MANDATORY)

Implement a pipeline that performs:

### Cleaning

* Remove duplicates
* Handle missing values
* Convert date columns
* Remove irrelevant columns (video_id, thumbnail_link, etc.)

### Feature Engineering

Create additional attributes so the dataset has ≥100 attributes:

From dates:

* day_of_week
* month
* trending_duration

Engagement metrics:

* likes_per_view
* comments_per_view
* engagement_rate

From tags column:

* number of tags
* TF-IDF representation

From title:

* title length
* sentiment score (simple approach)

One-hot encoding for:

* category_id

### Scaling

* StandardScaler or MinMaxScaler

Save the processed dataset to `/data/processed`.

---

## 2. Clustering (Minimum 5 Algorithms)

Implement the following algorithms:

1. KMeans
2. Agglomerative Clustering
3. DBSCAN
4. Gaussian Mixture Models
5. Spectral Clustering

For each algorithm:

* Train the model
* Save the model in `/models`
* Save cluster labels
* Use reasonable/optimal parameters (e.g., elbow method for KMeans)

---

## 3. Dimensionality Reduction

Build models for:

* All features
* PCA reduced dataset
* SelectKBest reduced dataset

Compare performance.

---

## 4. Evaluation

Use the following metrics:

* Silhouette score
* Davies-Bouldin index
* Calinski-Harabasz score

Create a comparison table of all algorithms.

---

## 5. Visualization (MANDATORY)

Create:

* 2D scatter plot (PCA projection)
* 3D scatter plot
* Cluster distribution plot
* Correlation heatmap
* Elbow curve for KMeans

---

## 6. Results Analysis

Automatically generate a textual summary describing:

* Which algorithm produces the best clusters
* Cluster interpretation
* Cluster characteristics
* Differences between models
* Impact of dimensionality reduction

Save as `.txt` or `.md`.

---

## 7. Main Script

`main.py` must:

1. Load raw data
2. Perform preprocessing
3. Perform feature engineering
4. Train all models
5. Evaluate models
6. Generate visualizations
7. Save results

---

## requirements.txt

Include at minimum:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
scipy
joblib
```

---

## README.md

Must include:

* Project description
* Dataset description
* How to run
* Project structure
* Algorithm descriptions
* How to reproduce results

---

## Mandatory Assignment Requirements

The following must be satisfied:

* Data preprocessing
* Minimum 5 algorithms
* 2D and 3D visualization
* Results analysis
* Models with full and reduced feature sets
* Text column processing
* Reproducibility
* Algorithm comparison

---

## Implementation Style

* Clean code
* Functions no longer than ~40 lines
* Clear docstrings
* Avoid unnecessary classes
* Focus on readability

---

## Required Outputs

* Processed dataset CSV
* Saved models
* Plots (PNG)
* Evaluation table CSV
* Summary analysis
* Reproducible pipeline

---

If needed, simplify the pipeline but do not remove any mandatory step.
