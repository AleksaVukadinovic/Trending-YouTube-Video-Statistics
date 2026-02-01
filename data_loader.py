import os
import glob
import json
import pandas as pd


class YouTubeDataLoader:
    """
    Handles loading of CSV data and JSON category mappings from the dataset path.
    """
    def __init__(self, path):
        self.path = path
        self.category_map = {}

    def load_categories(self):
        """Parsing JSON files to map category_id to actual category names."""
        json_files = glob.glob(os.path.join(self.path, "*.json"))
        for file in json_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    for item in data['items']:
                        cat_id = int(item['id'])
                        cat_title = item['snippet']['title']
                        self.category_map[cat_id] = cat_title
            except Exception as e:
                print(f"Warning: Could not parse {file}: {e}")

    def load_data(self):
        """Loads all CSV files and merges them into a single DataFrame."""
        csv_files = glob.glob(os.path.join(self.path, "*.csv"))
        all_dfs = []

        print(f"Found {len(csv_files)} regional files. Loading...")

        for file in csv_files:
            try:
                filename = os.path.basename(file)
                region = filename[:2].upper()

                df = pd.read_csv(file, encoding='latin1', on_bad_lines='skip')

                df['region'] = region
                all_dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")

        if not all_dfs:
            raise ValueError("No CSV files loaded.")

        full_df = pd.concat(all_dfs, ignore_index=True)
        return full_df
