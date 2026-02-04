import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import warnings

warnings.filterwarnings('ignore')


class AnomalyDetector:
    """
    Detect anomalies/outliers in the dataset using multiple methods.
    """

    def __init__(self, X_scaled, df=None, feature_names=None):
        self.X_scaled = X_scaled
        self.df = df
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X_scaled.shape[1])]
        self.anomaly_results = {}

    def isolation_forest(self, contamination=0.05, n_estimators=100):
        """Detect anomalies using Isolation Forest."""
        print(f"Running Isolation Forest (contamination={contamination})...")
        
        iso_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        
        predictions = iso_forest.fit_predict(self.X_scaled)
        scores = iso_forest.score_samples(self.X_scaled)
        
        
        anomalies = predictions == -1
        
        self.anomaly_results['isolation_forest'] = {
            'predictions': predictions,
            'scores': scores,
            'anomaly_mask': anomalies,
            'n_anomalies': anomalies.sum(),
            'model': iso_forest
        }
        
        print(f"Found {anomalies.sum()} anomalies ({100*anomalies.sum()/len(anomalies):.1f}%)")
        return anomalies, scores

    def local_outlier_factor(self, contamination=0.05, n_neighbors=20):
        """Detect anomalies using Local Outlier Factor."""
        print(f"Running Local Outlier Factor (contamination={contamination})...")
        
        lof = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors,
            n_jobs=-1
        )
        
        predictions = lof.fit_predict(self.X_scaled)
        scores = lof.negative_outlier_factor_
        
        anomalies = predictions == -1
        
        self.anomaly_results['lof'] = {
            'predictions': predictions,
            'scores': scores,
            'anomaly_mask': anomalies,
            'n_anomalies': anomalies.sum()
        }
        
        print(f"Found {anomalies.sum()} anomalies ({100*anomalies.sum()/len(anomalies):.1f}%)")
        return anomalies, scores

    def elliptic_envelope(self, contamination=0.05):
        """Detect anomalies using Elliptic Envelope (assumes Gaussian distribution)."""
        print(f"Running Elliptic Envelope (contamination={contamination})...")
        
        try:
            envelope = EllipticEnvelope(
                contamination=contamination,
                random_state=42
            )
            
            predictions = envelope.fit_predict(self.X_scaled)
            scores = envelope.score_samples(self.X_scaled)
            
            anomalies = predictions == -1
            
            self.anomaly_results['elliptic_envelope'] = {
                'predictions': predictions,
                'scores': scores,
                'anomaly_mask': anomalies,
                'n_anomalies': anomalies.sum(),
                'model': envelope
            }
            
            print(f"Found {anomalies.sum()} anomalies ({100*anomalies.sum()/len(anomalies):.1f}%)")
            return anomalies, scores
            
        except Exception as e:
            print(f"Elliptic Envelope failed: {e}")
            return None, None

    def one_class_svm(self, nu=0.05, kernel='rbf'):
        """Detect anomalies using One-Class SVM."""
        print(f"Running One-Class SVM (nu={nu})...")
        
        
        if len(self.X_scaled) > 10000:
            print("Sampling 10000 points for SVM training...")
            sample_idx = np.random.choice(len(self.X_scaled), 10000, replace=False)
            X_train = self.X_scaled[sample_idx]
        else:
            X_train = self.X_scaled
        
        svm = OneClassSVM(nu=nu, kernel=kernel)
        svm.fit(X_train)
        
        predictions = svm.predict(self.X_scaled)
        scores = svm.score_samples(self.X_scaled)
        
        anomalies = predictions == -1
        
        self.anomaly_results['one_class_svm'] = {
            'predictions': predictions,
            'scores': scores,
            'anomaly_mask': anomalies,
            'n_anomalies': anomalies.sum(),
            'model': svm
        }
        
        print(f"Found {anomalies.sum()} anomalies ({100*anomalies.sum()/len(anomalies):.1f}%)")
        return anomalies, scores

    def ensemble_anomalies(self, min_votes=2):
        """Combine results from multiple methods."""
        print(f"\nCreating ensemble (min_votes={min_votes})...")
        
        if len(self.anomaly_results) < 2:
            print("Need at least 2 methods for ensemble. Running all methods...")
            self.run_all_methods()
        
        
        votes = np.zeros(len(self.X_scaled))
        
        for method, result in self.anomaly_results.items():
            if method != 'ensemble':
                votes += result['anomaly_mask'].astype(int)
        
        ensemble_anomalies = votes >= min_votes
        
        self.anomaly_results['ensemble'] = {
            'votes': votes,
            'anomaly_mask': ensemble_anomalies,
            'n_anomalies': ensemble_anomalies.sum(),
            'min_votes': min_votes
        }
        
        print(f"Ensemble found {ensemble_anomalies.sum()} anomalies "
              f"({100*ensemble_anomalies.sum()/len(ensemble_anomalies):.1f}%)")
        
        return ensemble_anomalies, votes

    def run_all_methods(self, contamination=0.05):
        """Run all anomaly detection methods."""
        self.isolation_forest(contamination=contamination)
        self.local_outlier_factor(contamination=contamination)
        self.elliptic_envelope(contamination=contamination)
        
        if len(self.X_scaled) <= 20000:
            self.one_class_svm(nu=contamination)
        
        return self.anomaly_results

    def plot_anomaly_comparison(self):
        """Compare anomaly detection results across methods."""
        if len(self.anomaly_results) < 2:
            self.run_all_methods()
        
        methods = [m for m in self.anomaly_results.keys() if m != 'ensemble']
        n_methods = len(methods)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        
        ax1 = axes[0, 0]
        counts = [self.anomaly_results[m]['n_anomalies'] for m in methods]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))
        ax1.bar(methods, counts, color=colors)
        ax1.set_ylabel('Number of Anomalies')
        ax1.set_title('Anomaly Count by Method')
        ax1.tick_params(axis='x', rotation=45)
        
        
        ax2 = axes[0, 1]
        for method in methods:
            scores = self.anomaly_results[method]['scores']
            ax2.hist(scores, bins=50, alpha=0.5, label=method, density=True)
        ax2.set_xlabel('Anomaly Score')
        ax2.set_ylabel('Density')
        ax2.set_title('Anomaly Score Distributions')
        ax2.legend()
        
        
        ax3 = axes[1, 0]
        agreement_matrix = np.zeros((n_methods, n_methods))
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                mask1 = self.anomaly_results[m1]['anomaly_mask']
                mask2 = self.anomaly_results[m2]['anomaly_mask']
                agreement_matrix[i, j] = np.mean(mask1 == mask2)
        
        sns.heatmap(agreement_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=methods, yticklabels=methods, ax=ax3,
                   vmin=0.8, vmax=1.0)
        ax3.set_title('Method Agreement Matrix')
        
        
        ax4 = axes[1, 1]
        
        
        if 'ensemble' not in self.anomaly_results:
            self.ensemble_anomalies()
        
        votes = self.anomaly_results['ensemble']['votes']
        vote_counts = [np.sum(votes == i) for i in range(n_methods + 1)]
        
        ax4.bar(range(n_methods + 1), vote_counts, color=plt.cm.Reds(np.linspace(0.2, 0.8, n_methods + 1)))
        ax4.set_xlabel('Number of Methods Agreeing (Anomaly)')
        ax4.set_ylabel('Number of Samples')
        ax4.set_title('Agreement Distribution')
        ax4.set_xticks(range(n_methods + 1))
        
        plt.tight_layout()
        plt.show()

    def plot_anomalies_2d(self, labels=None, method='isolation_forest'):
        """Visualize anomalies in 2D using PCA."""
        from sklearn.decomposition import PCA
        
        if method not in self.anomaly_results:
            print(f"Method {method} not found. Running it first...")
            if method == 'isolation_forest':
                self.isolation_forest()
            elif method == 'lof':
                self.local_outlier_factor()
        
        anomaly_mask = self.anomaly_results[method]['anomaly_mask']
        scores = self.anomaly_results[method]['scores']
        
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        
        ax1 = axes[0]
        ax1.scatter(X_pca[~anomaly_mask, 0], X_pca[~anomaly_mask, 1], 
                   c='blue', alpha=0.3, s=10, label='Normal')
        ax1.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], 
                   c='red', alpha=0.8, s=30, label='Anomaly')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_title(f'Anomalies ({method})')
        ax1.legend()
        
        
        ax2 = axes[1]
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=scores, 
                             cmap='RdYlGn', alpha=0.6, s=10)
        plt.colorbar(scatter, ax=ax2, label='Anomaly Score')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('Anomaly Scores')
        
        
        ax3 = axes[2]
        if labels is not None:
            scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                                 cmap='viridis', alpha=0.6, s=10)
            
            ax3.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], 
                       facecolors='none', edgecolors='red', s=50, linewidths=2)
            plt.colorbar(scatter, ax=ax3, label='Cluster')
            ax3.set_title('Clusters with Anomalies Highlighted')
        else:
            ax3.text(0.5, 0.5, 'No cluster labels provided', 
                    ha='center', va='center', transform=ax3.transAxes)
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        
        plt.tight_layout()
        plt.show()

    def get_anomaly_details(self, method='isolation_forest', top_n=20):
        """Get details of top anomalies."""
        if method not in self.anomaly_results:
            print(f"Method {method} not found")
            return None
        
        scores = self.anomaly_results[method]['scores']
        anomaly_mask = self.anomaly_results[method]['anomaly_mask']
        
        
        anomaly_indices = np.where(anomaly_mask)[0]
        anomaly_scores = scores[anomaly_mask]
        
        
        sorted_idx = np.argsort(anomaly_scores)
        top_anomaly_indices = anomaly_indices[sorted_idx][:top_n]
        
        if self.df is not None:
            anomaly_df = self.df.iloc[top_anomaly_indices].copy()
            anomaly_df['anomaly_score'] = scores[top_anomaly_indices]
            return anomaly_df
        else:
            return pd.DataFrame({
                'index': top_anomaly_indices,
                'anomaly_score': scores[top_anomaly_indices]
            })

    def plot_anomaly_feature_analysis(self, method='isolation_forest', top_n=10):
        """Analyze which features contribute most to anomalies."""
        if method not in self.anomaly_results:
            self.isolation_forest()
        
        anomaly_mask = self.anomaly_results[method]['anomaly_mask']
        
        
        fig, axes = plt.subplots(2, min(5, len(self.feature_names)), 
                                figsize=(4*min(5, len(self.feature_names)), 8))
        
        n_features = min(top_n, len(self.feature_names))
        
        
        deviations = []
        for i, feature in enumerate(self.feature_names):
            normal_mean = self.X_scaled[~anomaly_mask, i].mean()
            anomaly_mean = self.X_scaled[anomaly_mask, i].mean()
            normal_std = self.X_scaled[~anomaly_mask, i].std()
            
            deviation = abs(anomaly_mean - normal_mean) / (normal_std + 1e-10)
            deviations.append((feature, deviation, i))
        
        
        deviations.sort(key=lambda x: x[1], reverse=True)
        
        for plot_idx, (feature, deviation, feat_idx) in enumerate(deviations[:n_features]):
            row = plot_idx // 5
            col = plot_idx % 5
            
            if len(self.feature_names) <= 5:
                ax = axes[row, col] if len(self.feature_names) > 1 else axes[col]
            else:
                ax = axes[row, col]
            
            ax.hist(self.X_scaled[~anomaly_mask, feat_idx], bins=30, alpha=0.5, 
                   label='Normal', density=True)
            ax.hist(self.X_scaled[anomaly_mask, feat_idx], bins=30, alpha=0.5, 
                   label='Anomaly', density=True)
            ax.set_title(f'{feature}\n(dev={deviation:.2f})')
            ax.legend(fontsize=8)
        
        plt.suptitle('Feature Distributions: Normal vs Anomaly', fontsize=14)
        plt.tight_layout()
        plt.show()
