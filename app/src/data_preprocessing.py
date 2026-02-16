import pandas as pd
import numpy as np
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_raw_data(file_path: str) -> pd.DataFrame:
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Loaded data with {encoding} encoding: {df.shape}")
            return df
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Could not read file with any supported encoding: {file_path}")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    initial_count = len(df)
    
    if 'video_id' in df.columns:
        df = df.drop_duplicates(subset=['video_id'], keep='first')
    else:
        df = df.drop_duplicates(keep='first')
    
    removed = initial_count - len(df)
    print(f"Removed {removed} duplicate rows. Remaining: {len(df)}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_before = df.isnull().sum().sum()
    
    for col in df.columns:
        if df[col].dtype in ['object', 'string', 'str'] or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].fillna('')
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    missing_after = df.isnull().sum().sum()
    print(f"Handled missing values: {missing_before} -> {missing_after}")
    return df


def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    date_columns = ['trending_date', 'publish_time']
    
    for col in date_columns:
        if col in df.columns:
            if col == 'trending_date':
                df[col] = pd.to_datetime(df[col], format='%y.%d.%m', errors='coerce')
            else:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"Converted {col} to datetime")
    
    return df


def remove_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_remove = [
        'video_id', 'thumbnail_link', 'description', 
        'video_error_or_removed', 'ratings_disabled', 'comments_disabled'
    ]
    
    existing_to_remove = [col for col in columns_to_remove if col in df.columns]
    df = df.drop(columns=existing_to_remove, errors='ignore')
    print(f"Removed columns: {existing_to_remove}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Starting Data Preprocessing ===")
    
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = convert_date_columns(df)
    df = remove_irrelevant_columns(df)
    
    df = df.reset_index(drop=True)
    print(f"\nPreprocessing complete. Final shape: {df.shape}")
    return df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to: {output_path}")
