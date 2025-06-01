from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier


# Model parameters
MODELS = {
    'svm': SVC(),
    'decision_tree': DecisionTreeClassifier(),
    'random_forest': RandomForestClassifier(),
    'xgboost': XGBClassifier()
}

# Data paths
DATA_PATH = "hand_landmarks_data.csv"

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Preprocessing
SCALER = StandardScaler()
ENCODER = LabelEncoder()