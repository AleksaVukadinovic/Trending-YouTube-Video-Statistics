import pandas as pd


class DataPreprocessor:
    """
    Handles cleaning, feature engineering, and preparation for analysis.
    """
    def __init__(self, df, category_map):
        self.df = df
        self.category_map = category_map

    def clean_and_engineer(self):
        print("Preprocessing data and engineering features...")
        self.df['category_name'] = self.df['category_id'].map(self.category_map).fillna("Unknown")
        self.df['trending_date'] = pd.to_datetime(self.df['trending_date'], format='%y.%d.%m', errors='coerce')
        self.df['publish_time'] = pd.to_datetime(self.df['publish_time'], errors='coerce')

        self.df.dropna(subset=['trending_date', 'publish_time'], inplace=True)

        publish_normalized = self.df['publish_time'].dt.tz_localize(None).dt.normalize()
        self.df['days_to_trend'] = (self.df['trending_date'] - publish_normalized).dt.days
        self.df['days_to_trend'] = self.df['days_to_trend'].apply(lambda x: x if x >= 0 else 0)

        self.df['title_length'] = self.df['title'].apply(lambda x: len(str(x)))
        self.df['tag_count'] = self.df['tags'].apply(lambda x: len(str(x).split('|')))

        self.df['like_view_ratio'] = self.df['likes'] / (self.df['views'] + 1)
        self.df['dislike_view_ratio'] = self.df['dislikes'] / (self.df['views'] + 1)
        self.df['comment_view_ratio'] = self.df['comment_count'] / (self.df['views'] + 1)
        self.df['engagement_rate'] = (self.df['likes'] + self.df['comment_count']) / (self.df['views'] + 1)
        self.df_unique = self.df.sort_values('views', ascending=False).drop_duplicates(subset=['video_id'], keep='first')
        print(f"Data cleaned. Total entries: {len(self.df)}. Unique videos: {len(self.df_unique)}")
        return self.df_unique
