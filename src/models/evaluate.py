"""Model evaluation utilities"""
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from mlflow.tracking import MlflowClient
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def get_best_model(experiment_name: str, metric: str = "test_roc_auc") -> Dict:
    """
    Retrieve the best model from MLflow based on a metric.
    
    Args:
        experiment_name: Name of the MLflow experiment
        metric: Metric to use for ranking (default: test_roc_auc)
        
    Returns:
        Dictionary with best model information
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    runs = client.search_runs(
        experiment.experiment_id,
        order_by=[f"metrics.{metric} DESC"]
    )
    
    if not runs:
        raise ValueError("No runs found in experiment")
    
    best_run = runs[0]
    best_model_info = {
        'run_id': best_run.info.run_id,
        'model_name': best_run.data.tags.get('mlflow.runName', best_run.info.run_id),
        'metrics': {k: v for k, v in best_run.data.metrics.items() if k.startswith('test_')},
        'params': best_run.data.params
    }
    
    logger.info(f"Best model: {best_model_info['model_name']}")
    return best_model_info


def get_model_comparison(experiment_name: str) -> pd.DataFrame:
    """
    Get comparison of all models in the experiment.
    
    Args:
        experiment_name: Name of the MLflow experiment
        
    Returns:
        DataFrame with model comparison
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    runs = client.search_runs(
        experiment.experiment_id,
        order_by=["metrics.test_roc_auc DESC"]
    )
    
    comparison_data = []
    for run in runs:
        run_data = {
            'Model': run.data.tags.get('mlflow.runName', run.info.run_id),
            'Test Accuracy': run.data.metrics.get('test_accuracy', 0),
            'Test Precision': run.data.metrics.get('test_precision', 0),
            'Test Recall': run.data.metrics.get('test_recall', 0),
            'Test F1-Score': run.data.metrics.get('test_f1_score', 0),
            'Test ROC-AUC': run.data.metrics.get('test_roc_auc', 0)
        }
        comparison_data.append(run_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df


def evaluate_model_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict:
    """
    Evaluate model predictions and return detailed metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with evaluation results
    """
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['No Diabetes', 'Diabetes'], output_dict=True)
    
    return {
        'confusion_matrix': cm,
        'classification_report': report
    }
