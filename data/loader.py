import pandas as pd
from config.pathes import DATA_DIR

def load_data(file_path=DATA_DIR / "hand_landmarks_data.csv"):
    """Load hand landmarks data"""
    return pd.read_csv(file_path)

def get_data_shape(df):
    """Return data shape"""
    return df.shape