import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance for cluster separation.
    """

    def __init__(self, X, labels, feature_names):
        self.X = X
        self.labels = labels
        self.feature_names = feature_names
        self.importance_results = {}

    def random_forest_importance(self, n_estimators=100):
        """Calculate feature importance using Random Forest classifier."""
        print("Calculating Random Forest feature importance...")
        
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rf.fit(self.X, self.labels)
        
        importances = rf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
        
        self.importance_results['random_forest'] = {
            'importances': importances,
            'std': std,
            'model': rf
        }
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances,
            'std': std
        }).sort_values('importance', ascending=False)

    def permutation_importance_analysis(self, n_repeats=10):
        """Calculate permutation importance."""
        print("Calculating permutation importance...")
        
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(self.X, self.labels)
        
        perm_importance = permutation_importance(
            rf, self.X, self.labels, 
            n_repeats=n_repeats, random_state=42, n_jobs=-1
        )
        
        self.importance_results['permutation'] = {
            'importances_mean': perm_importance.importances_mean,
            'importances_std': perm_importance.importances_std
        }
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False)

    def cluster_separation_analysis(self):
        """Analyze how well each feature separates clusters."""
        print("Analyzing cluster separation by feature...")
        
        separation_scores = []
        
        for i, feature in enumerate(self.feature_names):
            
            feature_values = self.X[:, i]
            
            
            total_var = np.var(feature_values)
            
            
            within_var = 0
            between_var = 0
            overall_mean = np.mean(feature_values)
            
            unique_labels = np.unique(self.labels)
            for label in unique_labels:
                mask = self.labels == label
                cluster_values = feature_values[mask]
                cluster_mean = np.mean(cluster_values)
                
                
                within_var += np.var(cluster_values) * len(cluster_values)
                
                
                between_var += len(cluster_values) * (cluster_mean - overall_mean) ** 2
            
            within_var /= len(feature_values)
            between_var /= len(feature_values)
            
            
            f_ratio = between_var / (within_var + 1e-10)
            
            separation_scores.append({
                'feature': feature,
                'f_ratio': f_ratio,
                'between_var': between_var,
                'within_var': within_var,
                'total_var': total_var
            })
        
        return pd.DataFrame(separation_scores).sort_values('f_ratio', ascending=False)

    def plot_feature_importance(self, top_n=15):
        """Visualize feature importance from multiple methods."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        
        if 'random_forest' not in self.importance_results:
            rf_df = self.random_forest_importance()
        else:
            rf_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.importance_results['random_forest']['importances'],
                'std': self.importance_results['random_forest']['std']
            }).sort_values('importance', ascending=False)
        
        ax1 = axes[0, 0]
        top_rf = rf_df.head(top_n)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_rf)))
        bars = ax1.barh(range(len(top_rf)), top_rf['importance'], xerr=top_rf['std'],
                       color=colors, alpha=0.8)
        ax1.set_yticks(range(len(top_rf)))
        ax1.set_yticklabels(top_rf['feature'])
        ax1.invert_yaxis()
        ax1.set_xlabel('Importance')
        ax1.set_title('Random Forest Feature Importance')
        
        
        if 'permutation' not in self.importance_results:
            perm_df = self.permutation_importance_analysis()
        else:
            perm_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.importance_results['permutation']['importances_mean'],
                'std': self.importance_results['permutation']['importances_std']
            }).sort_values('importance', ascending=False)
        
        ax2 = axes[0, 1]
        top_perm = perm_df.head(top_n)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_perm)))
        ax2.barh(range(len(top_perm)), top_perm['importance'], xerr=top_perm['std'],
                color=colors, alpha=0.8)
        ax2.set_yticks(range(len(top_perm)))
        ax2.set_yticklabels(top_perm['feature'])
        ax2.invert_yaxis()
        ax2.set_xlabel('Importance')
        ax2.set_title('Permutation Feature Importance')
        
        
        sep_df = self.cluster_separation_analysis()
        
        ax3 = axes[1, 0]
        top_sep = sep_df.head(top_n)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_sep)))
        ax3.barh(range(len(top_sep)), top_sep['f_ratio'], color=colors, alpha=0.8)
        ax3.set_yticks(range(len(top_sep)))
        ax3.set_yticklabels(top_sep['feature'])
        ax3.invert_yaxis()
        ax3.set_xlabel('F-Ratio')
        ax3.set_title('Cluster Separation Score (F-Ratio)')
        
        
        ax4 = axes[1, 1]
        
        
        rf_rank = rf_df.reset_index(drop=True)
        rf_rank['rf_rank'] = range(1, len(rf_rank) + 1)
        
        perm_rank = perm_df.reset_index(drop=True)
        perm_rank['perm_rank'] = range(1, len(perm_rank) + 1)
        
        sep_rank = sep_df.reset_index(drop=True)
        sep_rank['sep_rank'] = range(1, len(sep_rank) + 1)
        
        combined = rf_rank[['feature', 'rf_rank']].merge(
            perm_rank[['feature', 'perm_rank']], on='feature'
        ).merge(
            sep_rank[['feature', 'sep_rank']], on='feature'
        )
        combined['avg_rank'] = (combined['rf_rank'] + combined['perm_rank'] + combined['sep_rank']) / 3
        combined = combined.sort_values('avg_rank').head(top_n)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(combined)))
        ax4.barh(range(len(combined)), combined['avg_rank'].max() - combined['avg_rank'] + 1,
                color=colors, alpha=0.8)
        ax4.set_yticks(range(len(combined)))
        ax4.set_yticklabels(combined['feature'])
        ax4.invert_yaxis()
        ax4.set_xlabel('Combined Score (Higher = More Important)')
        ax4.set_title('Combined Feature Ranking')
        
        plt.tight_layout()
        plt.show()
        
        return combined

    def plot_feature_distributions_by_cluster(self, top_n=6):
        """Plot distributions of top features by cluster."""
        
        rf_df = self.random_forest_importance()
        top_features = rf_df.head(top_n)['feature'].tolist()
        
        n_cols = 3
        n_rows = (top_n + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features):
            ax = axes[i]
            feature_idx = self.feature_names.index(feature)
            
            data = pd.DataFrame({
                'value': self.X[:, feature_idx],
                'cluster': self.labels
            })
            
            for cluster in sorted(data['cluster'].unique()):
                cluster_data = data[data['cluster'] == cluster]['value']
                ax.hist(cluster_data, bins=30, alpha=0.5, label=f'Cluster {cluster}', density=True)
            
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {feature}')
            ax.legend()
        
        
        for i in range(top_n, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def plot_feature_correlation_with_clusters(self):
        """Plot correlation heatmap of features with cluster assignments."""
        
        cluster_dummies = pd.get_dummies(self.labels, prefix='cluster')
        
        
        feature_df = pd.DataFrame(self.X, columns=self.feature_names)
        combined = pd.concat([feature_df, cluster_dummies], axis=1)
        
        
        corr = combined.corr()
        
        
        cluster_cols = [col for col in corr.columns if col.startswith('cluster_')]
        cluster_corr = corr.loc[self.feature_names, cluster_cols]
        
        plt.figure(figsize=(12, max(8, len(self.feature_names) * 0.3)))
        sns.heatmap(cluster_corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f',
                   vmin=-1, vmax=1)
        plt.title('Feature-Cluster Correlation')
        plt.xlabel('Cluster')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
        
        return cluster_corr
