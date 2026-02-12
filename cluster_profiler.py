import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings('ignore')


class ClusterProfiler:
    """
    Generate comprehensive cluster profiles and business insights.
    """

    def __init__(self, df, labels, feature_cols, category_col='category_name'):
        self.df = df.copy()
        self.df['Cluster'] = labels
        self.labels = labels
        self.feature_cols = feature_cols
        self.category_col = category_col
        self.n_clusters = len(np.unique(labels[labels >= 0]))  

    def generate_cluster_summary(self):
        """Generate statistical summary for each cluster."""
        
        numeric_cols = ['views', 'likes', 'dislikes', 'comment_count', 'engagement_rate',
                       'days_to_trend', 'title_length', 'tag_count']
        numeric_cols = [c for c in numeric_cols if c in self.df.columns]
        
        summary = self.df.groupby('Cluster')[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max'])
        
        
        counts = self.df['Cluster'].value_counts().sort_index()
        
        return summary, counts

    def generate_cluster_profiles(self):
        """Generate human-readable profiles for each cluster."""
        profiles = {}
        
        
        overall_stats = {
            'views': self.df['views'].median(),
            'likes': self.df['likes'].median(),
            'engagement_rate': self.df['engagement_rate'].median() if 'engagement_rate' in self.df.columns else 0,
            'days_to_trend': self.df['days_to_trend'].median() if 'days_to_trend' in self.df.columns else 0
        }
        
        for cluster in range(self.n_clusters):
            cluster_df = self.df[self.df['Cluster'] == cluster]
            
            profile = {
                'size': len(cluster_df),
                'percentage': 100 * len(cluster_df) / len(self.df),
                'metrics': {}
            }
            
            
            for col in ['views', 'likes', 'dislikes', 'comment_count', 'engagement_rate', 'days_to_trend']:
                if col in cluster_df.columns:
                    profile['metrics'][col] = {
                        'mean': cluster_df[col].mean(),
                        'median': cluster_df[col].median(),
                        'vs_overall': cluster_df[col].median() / (overall_stats.get(col, 1) + 1e-10)
                    }
            
            
            if self.category_col in cluster_df.columns:
                profile['top_categories'] = cluster_df[self.category_col].value_counts().head(5).to_dict()
            
            
            profile['description'] = self._generate_description(profile, overall_stats)
            
            profiles[cluster] = profile
        
        return profiles

    def _generate_description(self, profile, overall_stats):
        """Generate a human-readable description for a cluster."""
        descriptions = []
        
        metrics = profile['metrics']
        
        
        if 'views' in metrics:
            views_ratio = metrics['views']['vs_overall']
            if views_ratio > 2:
                descriptions.append("Very high viewership (viral potential)")
            elif views_ratio > 1.2:
                descriptions.append("Above average viewership")
            elif views_ratio < 0.5:
                descriptions.append("Lower viewership (niche content)")
            else:
                descriptions.append("Average viewership")
        
        
        if 'engagement_rate' in metrics:
            eng_ratio = metrics['engagement_rate']['vs_overall']
            if eng_ratio > 1.5:
                descriptions.append("High engagement rate")
            elif eng_ratio < 0.7:
                descriptions.append("Low engagement rate")
        
        
        if 'days_to_trend' in metrics:
            days_ratio = metrics['days_to_trend']['vs_overall']
            if days_ratio < 0.5:
                descriptions.append("Quick to trend")
            elif days_ratio > 2:
                descriptions.append("Slow to trend (evergreen content)")
        
        return "; ".join(descriptions) if descriptions else "Standard content profile"

    def plot_cluster_dashboard(self, figsize=(20, 16)):
        """Create a comprehensive cluster dashboard."""
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        
        ax1 = fig.add_subplot(gs[0, 0])
        counts = self.df['Cluster'].value_counts().sort_index()
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(counts)))
        ax1.pie(counts.values, labels=[f'Cluster {i}\n({c:,})' for i, c in enumerate(counts.values)],
               colors=colors, autopct='%1.1f%%')
        ax1.set_title('Cluster Distribution', fontsize=12, fontweight='bold')
        
        
        ax2 = fig.add_subplot(gs[0, 1])
        self.df['log_views'] = np.log1p(self.df['views'])
        sns.boxplot(x='Cluster', y='log_views', data=self.df, ax=ax2, palette='viridis')
        ax2.set_title('Views Distribution by Cluster (Log Scale)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Log(Views)')
        
        
        ax3 = fig.add_subplot(gs[0, 2])
        if 'engagement_rate' in self.df.columns:
            sns.boxplot(x='Cluster', y='engagement_rate', data=self.df, ax=ax3, palette='viridis')
            ax3.set_title('Engagement Rate by Cluster', fontsize=12, fontweight='bold')
        
        
        ax4 = fig.add_subplot(gs[1, :2])
        if self.category_col in self.df.columns:
            cat_cluster = pd.crosstab(self.df[self.category_col], self.df['Cluster'], normalize='columns') * 100
            
            top_cats = self.df[self.category_col].value_counts().head(10).index
            cat_cluster_top = cat_cluster.loc[cat_cluster.index.isin(top_cats)]
            sns.heatmap(cat_cluster_top, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax4)
            ax4.set_title('Category Distribution by Cluster (%)', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Cluster')
        
        
        ax5 = fig.add_subplot(gs[1, 2])
        if 'days_to_trend' in self.df.columns:
            sns.violinplot(x='Cluster', y='days_to_trend', data=self.df, ax=ax5, palette='viridis')
            ax5.set_title('Days to Trend by Cluster', fontsize=12, fontweight='bold')
            ax5.set_ylim(0, self.df['days_to_trend'].quantile(0.95))
        
        
        ax6 = fig.add_subplot(gs[2, 0], projection='polar')
        self._plot_radar_chart(ax6)
        
        
        ax7 = fig.add_subplot(gs[2, 1:])
        ax7.axis('off')
        
        profiles = self.generate_cluster_profiles()
        profile_text = "CLUSTER PROFILES\n" + "="*50 + "\n\n"
        
        for cluster, profile in profiles.items():
            profile_text += f"Cluster {cluster}: {profile['description']}\n"
            profile_text += f"  • Size: {profile['size']:,} videos ({profile['percentage']:.1f}%)\n"
            if 'views' in profile['metrics']:
                profile_text += f"  • Median views: {profile['metrics']['views']['median']:,.0f}\n"
            if 'top_categories' in profile:
                top_cat = list(profile['top_categories'].keys())[0] if profile['top_categories'] else 'N/A'
                profile_text += f"  • Top category: {top_cat}\n"
            profile_text += "\n"
        
        ax7.text(0.05, 0.95, profile_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('YouTube Video Cluster Analysis Dashboard', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    def _plot_radar_chart(self, ax):
        """Create radar chart comparing cluster characteristics."""
        
        metrics = ['views', 'likes', 'engagement_rate', 'days_to_trend', 'title_length']
        metrics = [m for m in metrics if m in self.df.columns]
        
        if len(metrics) < 3:
            ax.text(0.5, 0.5, 'Insufficient metrics\nfor radar chart', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        
        normalized = {}
        for metric in metrics:
            cluster_means = self.df.groupby('Cluster')[metric].mean()
            min_val, max_val = cluster_means.min(), cluster_means.max()
            normalized[metric] = (cluster_means - min_val) / (max_val - min_val + 1e-10)
        
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, self.n_clusters))
        
        for cluster in range(self.n_clusters):
            values = [normalized[m][cluster] for m in metrics]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster}', color=colors[cluster])
            ax.fill(angles, values, alpha=0.1, color=colors[cluster])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=8)
        ax.set_title('Cluster Characteristics', fontsize=10, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

    def plot_cluster_evolution(self, time_col='trending_date'):
        """Plot how cluster distribution changes over time."""
        if time_col not in self.df.columns:
            print(f"Column {time_col} not found")
            return
        
        self.df[time_col] = pd.to_datetime(self.df[time_col])
        self.df['year_month'] = self.df[time_col].dt.to_period('M')
        
        
        time_cluster = pd.crosstab(self.df['year_month'], self.df['Cluster'], normalize='index') * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        
        ax1 = axes[0]
        time_cluster.plot(kind='area', stacked=True, ax=ax1, colormap='viridis', alpha=0.8)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Percentage')
        ax1.set_title('Cluster Distribution Over Time')
        ax1.legend(title='Cluster', bbox_to_anchor=(1.02, 1))
        ax1.tick_params(axis='x', rotation=45)
        
        
        ax2 = axes[1]
        time_cluster.plot(ax=ax2, marker='o', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Percentage')
        ax2.set_title('Cluster Trends Over Time')
        ax2.legend(title='Cluster', bbox_to_anchor=(1.02, 1))
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

    def export_cluster_report(self, filename='cluster_report.csv'):
        """Export cluster statistics to CSV."""
        summary, counts = self.generate_cluster_summary()
        
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary['count'] = counts
        
        summary.to_csv(filename)
        print(f"Report exported to {filename}")
        
        return summary

    def get_representative_videos(self, n_per_cluster=5):
        """Get representative videos from each cluster (closest to centroid)."""
        from sklearn.metrics import pairwise_distances
        
        representatives = {}
        
        
        feature_cols = [c for c in self.feature_cols if c in self.df.columns]
        X = self.df[feature_cols].values
        
        for cluster in range(self.n_clusters):
            mask = self.df['Cluster'] == cluster
            cluster_X = X[mask]
            cluster_df = self.df[mask]
            
            
            centroid = cluster_X.mean(axis=0).reshape(1, -1)
            
            
            distances = pairwise_distances(cluster_X, centroid).flatten()
            closest_idx = np.argsort(distances)[:n_per_cluster]
            
            representatives[cluster] = cluster_df.iloc[closest_idx]
        
        return representatives
