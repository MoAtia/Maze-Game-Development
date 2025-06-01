import joblib
from config.pathes import MODELS_DIR

def save_model(model, model_name):
    """Save trained model to disk"""
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / f"{model_name}.pkl")

def load_model(model_name):
    """Load trained model from disk"""
    return joblib.load(MODELS_DIR / f"{model_name}.pkl")