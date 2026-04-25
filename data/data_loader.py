# data/data_loader.py
import pandas as pd

def load_career_data(file_path):
    """Load AI Career recommendation dataset."""
    return pd.read_csv(file_path)
