import mlflow
import mlflow.sklearn
import mlflow.xgboost
from datetime import datetime
from config.pathes import MODELS_DIR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.metrics import ConfusionMatrixDisplay

def setup_mlflow():
    """Initialize MLflow tracking"""
    mlflow.set_tracking_uri("file:" + str(MODELS_DIR / "mlruns"))
    mlflow.set_experiment("HandGestureClassification")

def start_run(run_name=None):
    """Start MLflow run with optional name"""
    if not run_name:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return mlflow.start_run(run_name=run_name)

def log_params(params):
    """Log parameters dictionary"""
    mlflow.log_params(params)


def log_all_metrics(y_true, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='micro'),
        'recall': recall_score(y_true, y_pred, average='micro'),
        'f1': f1_score(y_true, y_pred,average='micro')
    }
    mlflow.log_metrics(metrics)



def log_confusion_matrix(y_true, y_pred):
    """Create and save a confusion matrix plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title("Confusion Matrix")
    temp_file = "confusion_matrix.png"
    fig.savefig(temp_file)
    plt.close(fig)
    mlflow.log_artifact(temp_file, artifact_path="confusion_matrix")


def log_model(model, model_name):
    """Log trained model"""
    mlflow.sklearn.log_model(model, model_name)


def end_run():
    """End current run"""
    mlflow.end_run()