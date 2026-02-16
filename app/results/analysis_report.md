============================================================
YOUTUBE TRENDING VIDEOS CLUSTERING ANALYSIS REPORT
Generated: 2026-02-16 23:44:51
============================================================

1. DATASET OVERVIEW
----------------------------------------
   - Total samples: 29,627
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

   Mainstream:
   - Size: 27,540 videos (93.0%)
   - Key characteristics (z-score from mean):

   Niche Music Viral:
   - Size: 79 videos (0.3%)
   - Key characteristics (z-score from mean):
     * Numeric Mean: +13.08σ
     * Views: +13.05σ
     * Numeric Max: +13.05σ
     * Numeric Range: +13.05σ
     * Numeric Std: +13.04σ

   Film Specialized:
   - Size: 1,883 videos (6.4%)
   - Key characteristics (z-score from mean):
     * Tfidf Tag 32: +2.77σ
     * Tfidf Tag 9: +2.62σ
     * Tfidf Tag 17: +2.57σ
     * Tfidf Tag 38: +2.26σ
     * Tfidf Tag 46: +1.97σ

   Niche People Specialized:
   - Size: 125 videos (0.4%)
   - Key characteristics (z-score from mean):
     * Tfidf Tag 3: +15.36σ
     * Tfidf Tag 33: +15.35σ
     * Tfidf Tag 37: +10.42σ
     * Category 22: +2.33σ
     * Title Has Numbers: +0.89σ

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