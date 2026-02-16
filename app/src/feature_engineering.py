"""
Feature engineering module for YouTube trending videos clustering.
Creates 100+ features from raw data including TF-IDF, engagement metrics, and temporal features.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from date columns."""
    df = df.copy()
    
    if 'trending_date' in df.columns and df['trending_date'].dtype == 'datetime64[ns]':
        df['trending_day_of_week'] = df['trending_date'].dt.dayofweek
        df['trending_month'] = df['trending_date'].dt.month
        df['trending_day'] = df['trending_date'].dt.day
        df['trending_is_weekend'] = (df['trending_day_of_week'] >= 5).astype(int)
    
    if 'publish_time' in df.columns and df['publish_time'].dtype == 'datetime64[ns]':
        df['publish_day_of_week'] = df['publish_time'].dt.dayofweek
        df['publish_month'] = df['publish_time'].dt.month
        df['publish_hour'] = df['publish_time'].dt.hour
        df['publish_day'] = df['publish_time'].dt.day
        df['publish_is_weekend'] = (df['publish_day_of_week'] >= 5).astype(int)
        
        if 'trending_date' in df.columns:
            df['trending_duration'] = (
                df['trending_date'] - df['publish_time']
            ).dt.total_seconds() / 86400
            df['trending_duration'] = df['trending_duration'].fillna(0).clip(lower=0)
    
    print(f"Created temporal features")
    return df


def create_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engagement-related features."""
    df = df.copy()
    
    views = df['views'].replace(0, 1) if 'views' in df.columns else 1
    
    if 'likes' in df.columns:
        df['likes_per_view'] = df['likes'] / views
        df['likes_log'] = np.log1p(df['likes'])
    
    if 'dislikes' in df.columns:
        df['dislikes_per_view'] = df['dislikes'] / views
        df['dislikes_log'] = np.log1p(df['dislikes'])
        
        if 'likes' in df.columns:
            total_votes = df['likes'] + df['dislikes']
            df['like_ratio'] = df['likes'] / total_votes.replace(0, 1)
            df['dislike_ratio'] = df['dislikes'] / total_votes.replace(0, 1)
    
    if 'comment_count' in df.columns:
        df['comments_per_view'] = df['comment_count'] / views
        df['comments_log'] = np.log1p(df['comment_count'])
    
    if 'views' in df.columns:
        df['views_log'] = np.log1p(df['views'])
    
    engagement_cols = ['likes', 'dislikes', 'comment_count']
    existing_cols = [c for c in engagement_cols if c in df.columns]
    if existing_cols and 'views' in df.columns:
        df['engagement_rate'] = df[existing_cols].sum(axis=1) / views
        df['total_interactions'] = df[existing_cols].sum(axis=1)
        df['interactions_log'] = np.log1p(df['total_interactions'])
    
    print(f"Created engagement features")
    return df


def create_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from text columns (title, tags)."""
    df = df.copy()
    
    if 'title' in df.columns:
        df['title_length'] = df['title'].astype(str).str.len()
        df['title_word_count'] = df['title'].astype(str).str.split().str.len()
        df['title_has_caps'] = df['title'].astype(str).str.isupper().astype(int)
        df['title_caps_ratio'] = df['title'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )
        df['title_has_numbers'] = df['title'].astype(str).str.contains(r'\d').astype(int)
        df['title_exclamation'] = df['title'].astype(str).str.count('!')
        df['title_question'] = df['title'].astype(str).str.count(r'\?')
        df['title_sentiment'] = df['title'].apply(simple_sentiment_score)
    
    if 'tags' in df.columns:
        df['tags_count'] = df['tags'].apply(count_tags)
        df['tags_length'] = df['tags'].astype(str).str.len()
        df['has_no_tags'] = (df['tags'] == '[none]').astype(int)
    
    if 'channel_title' in df.columns:
        df['channel_name_length'] = df['channel_title'].astype(str).str.len()
    
    print(f"Created text features")
    return df


def simple_sentiment_score(text: str) -> float:
    """Calculate a simple sentiment score based on positive/negative word presence."""
    text = str(text).lower()
    
    positive_words = [
        'best', 'amazing', 'awesome', 'great', 'excellent', 'fantastic', 
        'wonderful', 'love', 'perfect', 'beautiful', 'incredible', 'good',
        'happy', 'fun', 'exciting', 'brilliant', 'superb', 'outstanding'
    ]
    negative_words = [
        'worst', 'bad', 'terrible', 'horrible', 'awful', 'hate', 'ugly',
        'boring', 'stupid', 'fail', 'wrong', 'sad', 'angry', 'disappointed'
    ]
    
    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)
    
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total


def count_tags(tags_str: str) -> int:
    """Count number of tags in the tags string."""
    if pd.isna(tags_str) or tags_str == '[none]' or tags_str == '':
        return 0
    return len(str(tags_str).split('|'))


def create_tfidf_features(df: pd.DataFrame, max_features: int = 50) -> tuple:
    """Create TF-IDF features from tags column."""
    if 'tags' not in df.columns:
        return df, None
    
    tags_text = df['tags'].fillna('').astype(str).str.replace('|', ' ', regex=False)
    tags_text = tags_text.str.replace('[none]', '', regex=False)
    
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=5,
        max_df=0.95
    )
    
    tfidf_matrix = tfidf.fit_transform(tags_text)
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'tfidf_tag_{i}' for i in range(tfidf_matrix.shape[1])],
        index=df.index
    )
    
    df = pd.concat([df, tfidf_df], axis=1)
    print(f"Created {tfidf_matrix.shape[1]} TF-IDF features from tags")
    return df, tfidf


def create_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create one-hot encoded features for category_id."""
    df = df.copy()
    
    if 'category_id' in df.columns:
        category_dummies = pd.get_dummies(
            df['category_id'], 
            prefix='category',
            dtype=int
        )
        df = pd.concat([df, category_dummies], axis=1)
        print(f"Created {category_dummies.shape[1]} category one-hot features")
    
    return df


def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create statistical features from numerical columns."""
    df = df.copy()
    
    numeric_cols = ['views', 'likes', 'dislikes', 'comment_count']
    existing_numeric = [c for c in numeric_cols if c in df.columns]
    
    if len(existing_numeric) >= 2:
        df['numeric_std'] = df[existing_numeric].std(axis=1)
        df['numeric_mean'] = df[existing_numeric].mean(axis=1)
        df['numeric_max'] = df[existing_numeric].max(axis=1)
        df['numeric_min'] = df[existing_numeric].min(axis=1)
        df['numeric_range'] = df['numeric_max'] - df['numeric_min']
    
    print(f"Created statistical features")
    return df


def get_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only numeric features for clustering."""
    text_cols = ['title', 'channel_title', 'tags', 'trending_date', 'publish_time']
    cols_to_drop = [c for c in text_cols if c in df.columns]
    
    df_numeric = df.drop(columns=cols_to_drop, errors='ignore')
    df_numeric = df_numeric.select_dtypes(include=[np.number])
    
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
    df_numeric = df_numeric.fillna(df_numeric.median())
    
    print(f"Numeric features for clustering: {df_numeric.shape[1]}")
    return df_numeric


def scale_features(df: pd.DataFrame, scaler_type: str = 'standard') -> tuple:
    """Scale numeric features using StandardScaler or MinMaxScaler."""
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    print(f"Scaled features using {scaler_type} scaler")
    return scaled_df, scaler


def engineer_features(df: pd.DataFrame, tfidf_max_features: int = 50) -> tuple:
    """Run the complete feature engineering pipeline."""
    print("\n=== Starting Feature Engineering ===")
    
    df = create_temporal_features(df)
    df = create_engagement_features(df)
    df = create_text_features(df)
    df = create_statistical_features(df)
    df = create_category_features(df)
    df, tfidf_vectorizer = create_tfidf_features(df, max_features=tfidf_max_features)
    
    df_numeric = get_numeric_features(df)
    df_scaled, scaler = scale_features(df_numeric)
    
    print(f"\nFeature engineering complete. Final shape: {df_scaled.shape}")
    print(f"Total features: {df_scaled.shape[1]}")
    
    return df_scaled, scaler, tfidf_vectorizer, df_numeric.columns.tolist()
