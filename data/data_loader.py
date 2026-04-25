# data/data_loader.py
import pandas as pd

def load_career_data(file_path="data/raw/career_recommendation.csv"):
    """Load career recommendation dataset."""
    return pd.read_csv(file_path)