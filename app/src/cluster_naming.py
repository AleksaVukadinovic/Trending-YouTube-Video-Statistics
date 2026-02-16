import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


FEATURE_INTERPRETATIONS = {
    'views': ('Views', 'high views', 'low views'),
    'views_log': ('Views', 'high views', 'low views'),
    'likes': ('Likes', 'highly liked', 'few likes'),
    'likes_log': ('Likes', 'highly liked', 'few likes'),
    'likes_per_view': ('Like Rate', 'high like rate', 'low like rate'),
    'dislikes': ('Dislikes', 'controversial', 'non-controversial'),
    'dislikes_log': ('Dislikes', 'controversial', 'non-controversial'),
    'dislikes_per_view': ('Dislike Rate', 'high dislike rate', 'low dislike rate'),
    'like_ratio': ('Like Ratio', 'well-received', 'polarizing'),
    'dislike_ratio': ('Dislike Ratio', 'polarizing', 'well-received'),
    'comment_count': ('Comments', 'highly discussed', 'few comments'),
    'comments_log': ('Comments', 'highly discussed', 'few comments'),
    'comments_per_view': ('Comment Rate', 'high engagement', 'low engagement'),
    'engagement_rate': ('Engagement', 'high engagement', 'low engagement'),
    'total_interactions': ('Interactions', 'viral content', 'niche content'),
    'interactions_log': ('Interactions', 'viral content', 'niche content'),
    'title_length': ('Title Length', 'long titles', 'short titles'),
    'title_word_count': ('Title Words', 'descriptive titles', 'concise titles'),
    'title_caps_ratio': ('Title Caps', 'attention-grabbing', 'standard titles'),
    'title_exclamation': ('Exclamations', 'exciting content', 'calm content'),
    'title_question': ('Questions', 'curiosity-driven', 'statement-based'),
    'title_sentiment': ('Sentiment', 'positive sentiment', 'negative sentiment'),
    'title_has_numbers': ('Numbers in Title', 'list/ranking content', 'narrative content'),
    'tags_count': ('Tag Count', 'well-tagged', 'minimal tags'),
    'tags_length': ('Tags Length', 'detailed tagging', 'simple tagging'),
    'has_no_tags': ('No Tags', 'untagged', 'tagged'),
    'channel_name_length': ('Channel Name', 'branded channels', 'simple channels'),
    'trending_duration': ('Trending Duration', 'long-trending', 'quick-trending'),
    'trending_day_of_week': ('Trending Day', 'weekday trending', 'weekend trending'),
    'trending_is_weekend': ('Weekend Trending', 'weekend content', 'weekday content'),
    'publish_hour': ('Publish Hour', 'specific timing', 'varied timing'),
    'publish_is_weekend': ('Weekend Published', 'weekend uploads', 'weekday uploads'),
    'numeric_mean': ('Avg Metrics', 'high performance', 'moderate performance'),
    'numeric_max': ('Peak Metrics', 'viral peaks', 'steady performance'),
    'numeric_std': ('Metric Variance', 'variable performance', 'consistent performance'),
    'numeric_range': ('Metric Range', 'wide range', 'narrow range'),
}

CATEGORY_NAMES = {
    1: 'Film & Animation',
    2: 'Autos & Vehicles', 
    10: 'Music',
    15: 'Pets & Animals',
    17: 'Sports',
    18: 'Short Movies',
    19: 'Travel & Events',
    20: 'Gaming',
    21: 'Videoblogging',
    22: 'People & Blogs',
    23: 'Comedy',
    24: 'Entertainment',
    25: 'News & Politics',
    26: 'Howto & Style',
    27: 'Education',
    28: 'Science & Technology',
    29: 'Nonprofits & Activism',
    30: 'Movies',
    43: 'Shows',
    44: 'Trailers',
}


def analyze_cluster_characteristics(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    global_means: np.ndarray = None,
    global_stds: np.ndarray = None
) -> Dict[int, Dict]:
    if global_means is None:
        global_means = np.mean(data, axis=0)
    if global_stds is None:
        global_stds = np.std(data, axis=0)
        global_stds[global_stds == 0] = 1
    
    unique_labels = sorted(set(labels) - {-1})
    cluster_info = {}
    
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_data = data[mask]
        cluster_means = np.mean(cluster_data, axis=0)
        
        z_scores = (cluster_means - global_means) / global_stds
        
        feature_scores = list(zip(feature_names, z_scores))
        sorted_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)
        
        top_positive = [(f, z) for f, z in sorted_features[:10] if z > 0.5]
        top_negative = [(f, z) for f, z in sorted_features[-10:] if z < -0.5]
        top_negative.reverse()
        
        dominant_category = None
        category_features = [f for f in feature_names if f.startswith('category_') and f != 'category_id']
        if category_features:
            cat_indices = [feature_names.index(f) for f in category_features]
            cat_means = cluster_means[cat_indices]
            if len(cat_means) > 0 and np.max(cat_means) > global_means[cat_indices].mean() + 0.1:
                best_cat_idx = np.argmax(cat_means)
                best_cat_feature = category_features[best_cat_idx]
                try:
                    cat_id = int(best_cat_feature.replace('category_', ''))
                    dominant_category = CATEGORY_NAMES.get(cat_id, f'Category {cat_id}')
                except ValueError:
                    dominant_category = None
        
        cluster_info[cluster_id] = {
            'size': int(mask.sum()),
            'percentage': float(mask.sum() / len(labels) * 100),
            'top_positive_features': top_positive,
            'top_negative_features': top_negative,
            'dominant_category': dominant_category,
            'cluster_means': cluster_means,
            'z_scores': z_scores
        }
    
    return cluster_info


def generate_cluster_name(
    cluster_id: int,
    cluster_info: Dict,
    max_words: int = 4
) -> str:
    info = cluster_info[cluster_id]
    top_positive = info['top_positive_features']
    top_negative = info['top_negative_features']
    dominant_category = info['dominant_category']
    percentage = info['percentage']
    
    name_parts = []
    
    if percentage > 70:
        name_parts.append("Mainstream")
    elif percentage < 2:
        name_parts.append("Niche")
    
    if dominant_category:
        name_parts.append(dominant_category.split('&')[0].strip())
    
    engagement_features = ['engagement_rate', 'total_interactions', 'interactions_log', 
                          'comments_per_view', 'likes_per_view']
    viral_features = ['views', 'views_log', 'numeric_max', 'numeric_mean']
    
    for feat, z in top_positive[:3]:
        base_feat = feat.replace('_log', '').replace('tfidf_tag_', 'tag_')
        
        if base_feat in engagement_features and z > 1.5:
            name_parts.append("High-Engagement")
            break
        elif base_feat in viral_features and z > 2.0:
            name_parts.append("Viral")
            break
        elif 'like_ratio' in base_feat and z > 1.0:
            name_parts.append("Well-Liked")
            break
        elif 'dislike' in base_feat and z > 1.5:
            name_parts.append("Controversial")
            break
        elif 'comment' in base_feat and z > 1.5:
            name_parts.append("Discussion-Heavy")
            break
        elif 'title_length' in base_feat and z > 1.5:
            name_parts.append("Descriptive")
            break
        elif 'trending_duration' in base_feat and z > 1.5:
            name_parts.append("Long-Trending")
            break
        elif base_feat.startswith('tag_') and z > 2.0:
            name_parts.append("Specialized")
            break
    
    if len(name_parts) < 2:
        for feat, z in top_positive[:5]:
            if feat in FEATURE_INTERPRETATIONS:
                _, high_desc, _ = FEATURE_INTERPRETATIONS[feat]
                words = high_desc.split()
                if words[0].lower() not in ['high', 'low', 'few', 'many']:
                    name_parts.append(words[0].capitalize())
                else:
                    name_parts.append(high_desc.replace(' ', '-').title())
                break
    
    if not name_parts:
        if percentage > 50:
            name_parts = ["General", "Content"]
        else:
            name_parts = ["Mixed", "Content"]
    
    name = " ".join(name_parts[:max_words])
    
    if len(name) < 5:
        name = f"Group {cluster_id + 1}"
    
    return name


def generate_all_cluster_names(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str]
) -> Dict[int, str]:
    cluster_info = analyze_cluster_characteristics(data, labels, feature_names)
    
    cluster_names = {}
    used_names = set()
    
    for cluster_id in sorted(cluster_info.keys()):
        name = generate_cluster_name(cluster_id, cluster_info)
        
        if name in used_names:
            name = f"{name} {cluster_id + 1}"
        
        used_names.add(name)
        cluster_names[cluster_id] = name
    
    if -1 in set(labels):
        cluster_names[-1] = "Noise/Outliers"
    
    return cluster_names


def get_cluster_descriptions(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str]
) -> Dict[int, str]:
    cluster_info = analyze_cluster_characteristics(data, labels, feature_names)
    cluster_names = generate_all_cluster_names(data, labels, feature_names)
    
    descriptions = {}
    
    for cluster_id, info in cluster_info.items():
        name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        
        desc_parts = [
            f"**{name}** ({info['size']:,} videos, {info['percentage']:.1f}%)",
            ""
        ]
        
        if info['dominant_category']:
            desc_parts.append(f"Primary Category: {info['dominant_category']}")
        
        if info['top_positive_features']:
            desc_parts.append("Key Characteristics (above average):")
            for feat, z in info['top_positive_features'][:5]:
                readable_feat = feat.replace('_', ' ').replace('tfidf tag', 'Tag').title()
                desc_parts.append(f"  - {readable_feat}: {z:+.2f}σ")
        
        if info['top_negative_features']:
            desc_parts.append("Below Average In:")
            for feat, z in info['top_negative_features'][:3]:
                readable_feat = feat.replace('_', ' ').replace('tfidf tag', 'Tag').title()
                desc_parts.append(f"  - {readable_feat}: {z:+.2f}σ")
        
        descriptions[cluster_id] = "\n".join(desc_parts)
    
    return descriptions


def create_label_mapping(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str]
) -> Tuple[Dict[int, str], np.ndarray]:
    cluster_names = generate_all_cluster_names(data, labels, feature_names)
    
    string_labels = np.array([cluster_names.get(l, f"Cluster {l}") for l in labels])
    
    return cluster_names, string_labels
