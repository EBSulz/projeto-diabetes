"""Model training utilities"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {str(e)}")
            metrics['roc_auc'] = 0.0
    
    return metrics


def log_model_mlflow(
    model: Any,
    model_name: str,
    params: Dict[str, Any],
    metrics_train: Dict[str, float],
    metrics_test: Dict[str, float]
):
    """
    Log model, parameters, and metrics to MLflow.
    
    Args:
        model: Trained model
        model_name: Name of the model
        params: Model hyperparameters
        metrics_train: Training metrics
        metrics_test: Test metrics
    """
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)
        
        # Log training metrics
        for metric_name, metric_value in metrics_train.items():
            mlflow.log_metric(f"train_{metric_name}", metric_value)
        
        # Log test metrics
        for metric_name, metric_value in metrics_test.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        
        # Log model (using 'name' parameter instead of deprecated 'artifact_path')
        if isinstance(model, xgb.XGBClassifier):
            mlflow.xgboost.log_model(model, name="model")
        else:
            mlflow.sklearn.log_model(model, name="model")
        
        logger.info(f"{model_name} logged to MLflow")
        logger.info(f"Test metrics: {metrics_test}")


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict[str, Any] = None
) -> Tuple[LogisticRegression, Dict[str, float], Dict[str, float]]:
    """Train Logistic Regression model"""
    if params is None:
        params = {'C': 1.0, 'max_iter': 1000, 'random_state': 42}
    
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics_train = calculate_metrics(y_train, y_train_pred)
    metrics_test = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
    
    return model, metrics_train, metrics_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict[str, Any] = None
) -> Tuple[RandomForestClassifier, Dict[str, float], Dict[str, float]]:
    """Train Random Forest model"""
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics_train = calculate_metrics(y_train, y_train_pred)
    metrics_test = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
    
    return model, metrics_train, metrics_test


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict[str, Any] = None
) -> Tuple[xgb.XGBClassifier, Dict[str, float], Dict[str, float]]:
    """Train XGBoost model"""
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics_train = calculate_metrics(y_train, y_train_pred)
    metrics_test = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
    
    return model, metrics_train, metrics_test


def train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict[str, Any] = None
) -> Tuple[SVC, Dict[str, float], Dict[str, float]]:
    """Train SVM model"""
    if params is None:
        params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'random_state': 42,
            'probability': True
        }
    
    model = SVC(**params)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics_train = calculate_metrics(y_train, y_train_pred)
    metrics_test = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
    
    return model, metrics_train, metrics_test


def train_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict[str, Any] = None
) -> Tuple[KNeighborsClassifier, Dict[str, float], Dict[str, float]]:
    """Train KNN model"""
    if params is None:
        params = {
            'n_neighbors': 5,
            'weights': 'distance',
            'algorithm': 'auto'
        }
    
    model = KNeighborsClassifier(**params)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics_train = calculate_metrics(y_train, y_train_pred)
    metrics_test = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
    
    return model, metrics_train, metrics_test
