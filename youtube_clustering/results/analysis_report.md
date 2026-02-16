============================================================
YOUTUBE TRENDING VIDEOS CLUSTERING ANALYSIS REPORT
Generated: 2026-02-16 23:17:53
============================================================

1. DATASET OVERVIEW
----------------------------------------
   - Total samples: 29627
   - Total features: 102
   - Feature types: Numeric (scaled)

2. ALGORITHMS EVALUATED
----------------------------------------
   - KMeans
   - Agglomerative Clustering
   - DBSCAN
   - Gaussian Mixture Models (GMM)
   - Spectral Clustering

3. FEATURE SET VARIANTS
----------------------------------------
   - Full features (all engineered features)
   - PCA reduced (95% variance retained)
   - SelectKBest reduced (top 50 features)

4. EVALUATION METRICS
----------------------------------------

   FULL FEATURES:
   kmeans          | Silhouette:  0.0665 | DB:  2.6497 | CH:    1093.79
   agglomerative   | Silhouette:  0.2583 | DB:  1.8254 | CH:     976.47
   dbscan          | Silhouette:  0.3617 | DB:  0.7561 | CH:     365.16
   gmm             | Silhouette:  0.0101 | DB:  5.8799 | CH:     458.20
   spectral        | Silhouette:  0.1658 | DB:  2.5199 | CH:     487.54

   PCA FEATURES:
   kmeans          | Silhouette:  0.0698 | DB:  2.8620 | CH:    1231.70
   agglomerative   | Silhouette:  0.2406 | DB:  1.6159 | CH:    1020.38
   dbscan          | Silhouette:  0.3386 | DB:  0.7303 | CH:     333.94
   gmm             | Silhouette:  0.0263 | DB:  5.1608 | CH:     565.49
   spectral        | Silhouette:  0.2263 | DB:  2.0722 | CH:     277.81

   SELECTKBEST FEATURES:
   kmeans          | Silhouette:  0.1252 | DB:  2.0835 | CH:    2633.56
   agglomerative   | Silhouette:  0.0631 | DB:  2.4245 | CH:    2094.72
   dbscan          | Silhouette:  0.1118 | DB:  0.9453 | CH:     306.84
   gmm             | Silhouette:  0.1392 | DB:  3.0139 | CH:    1542.72
   spectral        | Silhouette:  0.2226 | DB:  1.3431 | CH:    1334.76

5. BEST PERFORMING MODEL
----------------------------------------
   Algorithm: AGGLOMERATIVE
   Feature Set: full
   Silhouette Score: 0.2583
   Davies-Bouldin Index: 1.8254
   Calinski-Harabasz Score: 976.47

6. CLUSTER INTERPRETATION
----------------------------------------

   Cluster 0:
   - Size: 27540 samples (93.0%)
   - Distinguishing features:
     * title_length: 0.069 (global mean: -0.000)
     * title_word_count: 0.057 (global mean: -0.000)
     * likes_per_view: 0.052 (global mean: 0.000)

   Cluster 1:
   - Size: 79 samples (0.3%)
   - Distinguishing features:
     * numeric_mean: 13.078 (global mean: -0.000)
     * views: 13.053 (global mean: 0.000)
     * numeric_max: 13.053 (global mean: 0.000)

   Cluster 2:
   - Size: 1883 samples (6.4%)
   - Distinguishing features:
     * tfidf_tag_32: 2.767 (global mean: 0.000)
     * tfidf_tag_9: 2.617 (global mean: -0.000)
     * tfidf_tag_17: 2.568 (global mean: 0.000)

   Cluster 3:
   - Size: 125 samples (0.4%)
   - Distinguishing features:
     * tfidf_tag_3: 15.363 (global mean: 0.000)
     * tfidf_tag_33: 15.351 (global mean: -0.000)
     * tfidf_tag_37: 10.425 (global mean: 0.000)

7. DIMENSIONALITY REDUCTION IMPACT
----------------------------------------
   KMEANS: PCA improved silhouette by 0.0033
   AGGLOMERATIVE: PCA decreased silhouette by 0.0177
   GMM: PCA improved silhouette by 0.0162

8. RECOMMENDATIONS
----------------------------------------
   - Use AGGLOMERATIVE with full features for best results
   - Consider the silhouette score for cluster quality assessment
   - Review cluster distributions for balanced groupings

============================================================
END OF REPORT
============================================================