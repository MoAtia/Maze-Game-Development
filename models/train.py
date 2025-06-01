from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from config.params import MODELS

def train_models(X_train, y_train):
    """Train multiple models and return trained models"""
    trained_models = {}
    
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models