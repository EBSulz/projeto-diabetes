"""Tests for model training"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.train import (
    calculate_metrics,
    train_logistic_regression,
    train_random_forest
)


@pytest.fixture
def sample_train_test_data():
    """Create sample train/test data"""
    np.random.seed(42)
    n_samples = 100
    
    X_train = pd.DataFrame({
        'Peso': np.random.randint(50, 150, n_samples),
        'Altura': np.random.randint(150, 200, n_samples),
        'BMI': np.random.uniform(20, 40, n_samples),
        'Hair_Preto': np.random.randint(0, 2, n_samples),
        'Hair_Loiro': np.random.randint(0, 2, n_samples)
    })
    
    y_train = pd.Series(np.random.randint(0, 2, n_samples))
    
    X_test = X_train.iloc[:20].copy()
    y_test = y_train.iloc[:20].copy()
    
    return X_train, y_train, X_test, y_test


def test_calculate_metrics(sample_train_test_data):
    """Test metrics calculation"""
    _, y_train, _, y_test = sample_train_test_data
    
    y_pred = np.random.randint(0, 2, len(y_test))
    y_pred_proba = np.random.rand(len(y_test))
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'roc_auc' in metrics
    
    # Check that metrics are between 0 and 1
    for metric_name, metric_value in metrics.items():
        assert 0 <= metric_value <= 1


def test_train_logistic_regression(sample_train_test_data):
    """Test logistic regression training"""
    X_train, y_train, X_test, y_test = sample_train_test_data
    
    model, metrics_train, metrics_test = train_logistic_regression(
        X_train, y_train, X_test, y_test
    )
    
    assert model is not None
    assert 'accuracy' in metrics_train
    assert 'accuracy' in metrics_test
    assert metrics_test['accuracy'] >= 0
    assert metrics_test['accuracy'] <= 1


def test_train_random_forest(sample_train_test_data):
    """Test random forest training"""
    X_train, y_train, X_test, y_test = sample_train_test_data
    
    model, metrics_train, metrics_test = train_random_forest(
        X_train, y_train, X_test, y_test
    )
    
    assert model is not None
    assert 'accuracy' in metrics_train
    assert 'accuracy' in metrics_test
    assert metrics_test['accuracy'] >= 0
    assert metrics_test['accuracy'] <= 1
