import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings

warnings.filterwarnings('ignore')


try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Note: vaderSentiment not installed. Install with: pip install vaderSentiment")


try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


class FeatureEngineer:
    """
    Advanced feature engineering including TF-IDF, temporal features, and sentiment analysis.
    """

    def __init__(self, df):
        self.df = df.copy()
        self.tfidf_vectorizer = None
        self.tfidf_features = None
        self.sentiment_analyzer = None
        
        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def add_temporal_features(self):
        """Extract temporal features from publish_time."""
        print("Engineering temporal features...")
        
        if 'publish_time' not in self.df.columns:
            print("Warning: publish_time column not found")
            return self.df
        
        
        self.df['publish_time'] = pd.to_datetime(self.df['publish_time'], errors='coerce')
        
        
        self.df['publish_hour'] = self.df['publish_time'].dt.hour
        
        
        self.df['publish_dayofweek'] = self.df['publish_time'].dt.dayofweek
        
        
        self.df['is_weekend'] = (self.df['publish_dayofweek'] >= 5).astype(int)
        
        
        self.df['publish_month'] = self.df['publish_time'].dt.month
        
        
        self.df['publish_quarter'] = self.df['publish_time'].dt.quarter
        
        
        def categorize_hour(hour):
            if pd.isna(hour):
                return 'unknown'
            hour = int(hour)
            if 5 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 21:
                return 'evening'
            else:
                return 'night'
        
        self.df['time_of_day'] = self.df['publish_hour'].apply(categorize_hour)
        
        
        self.df['day_name'] = self.df['publish_time'].dt.day_name()
        
        print(f"Added temporal features: publish_hour, publish_dayofweek, is_weekend, "
              f"publish_month, publish_quarter, time_of_day, day_name")
        
        return self.df

    def add_tfidf_features(self, text_column='title', max_features=100, n_components=20):
        """
        Extract TF-IDF features from text and reduce dimensions with SVD.
        """
        print(f"Extracting TF-IDF features from '{text_column}'...")
        
        if text_column not in self.df.columns:
            print(f"Warning: {text_column} column not found")
            return self.df, None
        
        
        texts = self.df[text_column].fillna('').astype(str)
        texts = texts.apply(self._clean_text)
        
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        
        n_components = min(n_components, tfidf_matrix.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        tfidf_reduced = svd.fit_transform(tfidf_matrix)
        
        
        tfidf_cols = [f'tfidf_{i}' for i in range(n_components)]
        tfidf_df = pd.DataFrame(tfidf_reduced, columns=tfidf_cols, index=self.df.index)
        self.df = pd.concat([self.df, tfidf_df], axis=1)
        
        self.tfidf_features = tfidf_cols
        
        print(f"Added {n_components} TF-IDF components (explaining {svd.explained_variance_ratio_.sum()*100:.1f}% variance)")
        
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        self.top_terms_per_component = {}
        for i, component in enumerate(svd.components_):
            top_indices = component.argsort()[-10:][::-1]
            self.top_terms_per_component[i] = [feature_names[idx] for idx in top_indices]
        
        return self.df, self.tfidf_features

    def add_tag_tfidf_features(self, max_features=50, n_components=10):
        """Extract TF-IDF features from tags."""
        print("Extracting TF-IDF features from tags...")
        
        if 'tags' not in self.df.columns:
            print("Warning: tags column not found")
            return self.df, None
        
        
        tags = self.df['tags'].fillna('').astype(str)
        tags = tags.apply(lambda x: ' '.join(x.split('|')))
        tags = tags.apply(self._clean_text)
        
        
        tag_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=3,
            max_df=0.95
        )
        
        try:
            tag_tfidf = tag_vectorizer.fit_transform(tags)
            
            n_components = min(n_components, tag_tfidf.shape[1] - 1)
            if n_components < 1:
                print("Not enough tag features for SVD")
                return self.df, None
                
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            tag_reduced = svd.fit_transform(tag_tfidf)
            
            tag_cols = [f'tag_tfidf_{i}' for i in range(n_components)]
            tag_df = pd.DataFrame(tag_reduced, columns=tag_cols, index=self.df.index)
            self.df = pd.concat([self.df, tag_df], axis=1)
            
            print(f"Added {n_components} tag TF-IDF components")
            return self.df, tag_cols
            
        except Exception as e:
            print(f"Error extracting tag features: {e}")
            return self.df, None

    def add_sentiment_features(self, text_column='title'):
        """Add sentiment analysis features using VADER or TextBlob."""
        print(f"Analyzing sentiment of '{text_column}'...")
        
        if text_column not in self.df.columns:
            print(f"Warning: {text_column} column not found")
            return self.df
        
        texts = self.df[text_column].fillna('').astype(str)
        
        if VADER_AVAILABLE and self.sentiment_analyzer:
            
            sentiments = texts.apply(lambda x: self.sentiment_analyzer.polarity_scores(x))
            self.df['sentiment_neg'] = sentiments.apply(lambda x: x['neg'])
            self.df['sentiment_neu'] = sentiments.apply(lambda x: x['neu'])
            self.df['sentiment_pos'] = sentiments.apply(lambda x: x['pos'])
            self.df['sentiment_compound'] = sentiments.apply(lambda x: x['compound'])
            
            
            def categorize_sentiment(compound):
                if compound >= 0.05:
                    return 'positive'
                elif compound <= -0.05:
                    return 'negative'
                else:
                    return 'neutral'
            
            self.df['sentiment_category'] = self.df['sentiment_compound'].apply(categorize_sentiment)
            print("Added VADER sentiment features: neg, neu, pos, compound, category")
            
        elif TEXTBLOB_AVAILABLE:
            
            self.df['sentiment_polarity'] = texts.apply(lambda x: TextBlob(x).sentiment.polarity)
            self.df['sentiment_subjectivity'] = texts.apply(lambda x: TextBlob(x).sentiment.subjectivity)
            
            def categorize_sentiment(polarity):
                if polarity > 0.1:
                    return 'positive'
                elif polarity < -0.1:
                    return 'negative'
                else:
                    return 'neutral'
            
            self.df['sentiment_category'] = self.df['sentiment_polarity'].apply(categorize_sentiment)
            print("Added TextBlob sentiment features: polarity, subjectivity, category")
            
        else:
            
            print("No sentiment library available. Using basic keyword matching...")
            
            positive_words = set(['amazing', 'awesome', 'best', 'beautiful', 'brilliant', 'excellent',
                                 'fantastic', 'great', 'incredible', 'love', 'perfect', 'wonderful'])
            negative_words = set(['awful', 'bad', 'boring', 'fail', 'hate', 'horrible', 'terrible',
                                 'worst', 'disappointing', 'sad', 'angry', 'scary'])
            
            def basic_sentiment(text):
                text_lower = text.lower()
                words = set(re.findall(r'\b\w+\b', text_lower))
                pos_count = len(words & positive_words)
                neg_count = len(words & negative_words)
                
                if pos_count > neg_count:
                    return 1
                elif neg_count > pos_count:
                    return -1
                else:
                    return 0
            
            self.df['sentiment_basic'] = texts.apply(basic_sentiment)
            self.df['sentiment_category'] = self.df['sentiment_basic'].map({
                1: 'positive', 0: 'neutral', -1: 'negative'
            })
            print("Added basic sentiment features: sentiment_basic, sentiment_category")
        
        return self.df

    def add_text_statistics(self, text_column='title'):
        """Add text statistics features."""
        print(f"Computing text statistics for '{text_column}'...")
        
        if text_column not in self.df.columns:
            return self.df
        
        texts = self.df[text_column].fillna('').astype(str)
        
        
        self.df[f'{text_column}_char_count'] = texts.apply(len)
        
        
        self.df[f'{text_column}_word_count'] = texts.apply(lambda x: len(x.split()))
        
        
        def avg_word_length(text):
            words = text.split()
            if not words:
                return 0
            return np.mean([len(w) for w in words])
        
        self.df[f'{text_column}_avg_word_len'] = texts.apply(avg_word_length)
        
        
        def uppercase_ratio(text):
            if len(text) == 0:
                return 0
            return sum(1 for c in text if c.isupper()) / len(text)
        
        self.df[f'{text_column}_uppercase_ratio'] = texts.apply(uppercase_ratio)
        
        
        self.df[f'{text_column}_punct_count'] = texts.apply(
            lambda x: sum(1 for c in x if c in '!?.,;:'))
        
        
        self.df[f'{text_column}_exclamation_count'] = texts.apply(lambda x: x.count('!'))
        
        
        self.df[f'{text_column}_question_count'] = texts.apply(lambda x: x.count('?'))
        
        
        self.df[f'{text_column}_has_number'] = texts.apply(
            lambda x: 1 if re.search(r'\d', x) else 0)
        
        
        self.df[f'{text_column}_caps_words'] = texts.apply(
            lambda x: sum(1 for w in x.split() if w.isupper() and len(w) > 1))
        
        print(f"Added text statistics for {text_column}")
        return self.df

    def _clean_text(self, text):
        """Clean text for TF-IDF processing."""
        if pd.isna(text):
            return ''
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_all_new_features(self):
        """Return list of all engineered feature names."""
        temporal_features = ['publish_hour', 'publish_dayofweek', 'is_weekend', 
                            'publish_month', 'publish_quarter']
        
        sentiment_features = []
        if 'sentiment_compound' in self.df.columns:
            sentiment_features = ['sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound']
        elif 'sentiment_polarity' in self.df.columns:
            sentiment_features = ['sentiment_polarity', 'sentiment_subjectivity']
        elif 'sentiment_basic' in self.df.columns:
            sentiment_features = ['sentiment_basic']
        
        text_stats = [col for col in self.df.columns if 'char_count' in col or 
                     'word_count' in col or 'avg_word_len' in col or
                     'uppercase_ratio' in col or 'punct_count' in col or
                     'exclamation_count' in col or 'question_count' in col or
                     'has_number' in col or 'caps_words' in col]
        
        tfidf_features = self.tfidf_features or []
        
        return {
            'temporal': temporal_features,
            'sentiment': sentiment_features,
            'text_stats': text_stats,
            'tfidf': tfidf_features
        }

    def engineer_all_features(self):
        """Run all feature engineering steps."""
        self.add_temporal_features()
        self.add_tfidf_features()
        self.add_tag_tfidf_features()
        self.add_sentiment_features()
        self.add_text_statistics()
        
        return self.df
