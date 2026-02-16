import kagglehub
from pathlib import Path

def download_dataset(dataset: str = "datasnaek/youtube-new") -> str:
    print(f"Downloading dataset from Kaggle: {dataset}")
    path = kagglehub.dataset_download(dataset)
    print(f"Dataset downloaded to: {path}")
    return path

def find_csv_file(dataset_dir: str) -> str:
    dataset_path = Path(dataset_dir)
    csv_files = list(dataset_path.glob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
    
    preferred = ['DEvideos.csv', 'USvideos.csv', 'GBvideos.csv']
    for pref in preferred:
        for csv in csv_files:
            if csv.name == pref:
                return str(csv)
    
    return str(csv_files[0])


def ensure_dataset() -> str:
    dataset_dir = download_dataset()
    return find_csv_file(dataset_dir)
