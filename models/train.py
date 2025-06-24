from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from config.params import MODELS

from utils.tracking import start_run, log_params, log_all_metrics, log_model, log_confusion_matrix, end_run

from config.params import MODELS, RANDOM_STATE, TEST_SIZE

import mlflow

def train_models(X_train, y_train, X_test, y_test, track_experiment=True):
    """Train and evaluate models with MLflow tracking"""
    trained_models = {}
    results = {}
    
    # Common parameters to log
    base_params = {
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE
    }
    
    for name, model in MODELS.items():
        if track_experiment:
            start_run(f"{name}_training")
            log_params({
                **base_params,
                'model_type': name,
                **model.get_params()  # Log model hyperparameters
            })
        
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy
        }
        
        if track_experiment:
            # Log metrics and model
            log_all_metrics(y_test, y_pred)
            log_confusion_matrix(y_test, y_pred)
            log_model(model, name)

            tags = {
                "model_family": name,
                "feature_set": "landmarks",
            }
            mlflow.set_tags(tags)

            end_run()
    
    return trained_models, results