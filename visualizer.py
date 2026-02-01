import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    """
    Generates plots for Visual Analysis.
    """
    def __init__(self, df):
        self.df = df

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(10, 8))
        cols = ['views', 'likes', 'dislikes', 'comment_count', 'days_to_trend', 'title_length', 'engagement_rate']
        corr = self.df[cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix of Key Metrics")
        plt.tight_layout()
        plt.show()

    def plot_category_popularity(self):
        plt.figure(figsize=(14, 6))
        order = self.df['category_name'].value_counts().index
        sns.countplot(data=self.df, x='category_name', order=order, palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.title("Number of Trending Videos per Category (Global)")
        plt.tight_layout()
        plt.show()

    def plot_engagement_by_region(self):
        plt.figure(figsize=(12, 6))
        
        self.df['log_views'] = np.log1p(self.df['views'])
        sns.boxplot(x='region', y='log_views', data=self.df, palette='Set3')
        plt.title("Distribution of Views (Log Scale) by Region")
        plt.ylabel("Log(Views)")
        plt.show()
