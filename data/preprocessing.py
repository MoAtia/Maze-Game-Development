from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Centering by wrist, and scaling by middle finger tip
def normalize_landmarks(row):
    coords = row.values.reshape(-1, 3)
    wrist = coords[0][:2]  # x, y of wrist
    mid_tip = coords[12][:2]  # x, y of middle finger tip
    scale = np.linalg.norm(mid_tip - wrist)
    coords[:, :2] = (coords[:, :2] - wrist) / (scale + 1e-6)
    return coords.flatten()


def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocess data: split, scale, encode"""
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Scale features
    normalized_data = X.apply(normalize_landmarks, axis=1, result_type='expand')

    # Ensure columns are named correctly
    normalized_data.columns = X.columns


    # Removing the wirst point
    normalized_data = normalized_data.drop(["x1","y1","z1"], axis=1)

    # after normalization, Zs components have zero mean and std as shown above in data description.
    normalized_data = normalized_data.drop([f"z{i}" for i in range(2,22)], axis=1)


    # Split data
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(normalized_data, y_encoded, test_size=test_size, random_state=random_state)    
    


    return X_train_scaled, X_test_scaled, y_train, y_test, encoder
