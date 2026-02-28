"""Tests for data loading and preprocessing"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.load_data import load_dataset, validate_data
from src.data.preprocessing import engineer_features, encode_categorical_features, ScalerManager


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data = {
        'id': [1, 2, 3, 4, 5],
        'Cor do cabelo': ['Preto', 'Loiro', 'Castanho', 'Preto', 'Careca'],
        'Peso': [70, 80, 65, 90, 75],
        'Altura': [170, 180, 165, 175, 172],
        'Diabético': [0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)


def test_engineer_features(sample_data):
    """Test feature engineering"""
    df_processed = engineer_features(sample_data)
    
    assert 'BMI' in df_processed.columns
    assert 'id' not in df_processed.columns
    assert len(df_processed) == len(sample_data)
    
    # Check BMI calculation
    expected_bmi = sample_data['Peso'] / ((sample_data['Altura'] / 100) ** 2)
    np.testing.assert_array_almost_equal(df_processed['BMI'].values, expected_bmi.values)


def test_encode_categorical_features(sample_data):
    """Test categorical encoding"""
    df_processed = engineer_features(sample_data)
    df_encoded = encode_categorical_features(df_processed)
    
    # Check that hair color columns are created
    hair_cols = [col for col in df_encoded.columns if col.startswith('Hair_')]
    assert len(hair_cols) > 0
    
    # Check that original categorical column is removed
    assert 'Cor do cabelo' not in df_encoded.columns


def test_scaler_manager(sample_data):
    """Test ScalerManager"""
    df_processed = engineer_features(sample_data)
    df_encoded = encode_categorical_features(df_processed)
    X = df_encoded.drop('Diabético', axis=1)
    
    scaler = ScalerManager()
    X_scaled = scaler.fit_transform(X)
    
    assert X_scaled.shape == X.shape
    assert isinstance(X_scaled, pd.DataFrame)
    
    # Check that scaler is fitted
    assert scaler.is_fitted


def test_validate_data(sample_data):
    """Test data validation"""
    # Should pass with valid data
    assert validate_data(sample_data) is True
    
    # Should fail with missing columns
    invalid_data = sample_data.drop('Peso', axis=1)
    with pytest.raises(ValueError):
        validate_data(invalid_data)
