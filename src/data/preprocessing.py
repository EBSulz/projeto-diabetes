"""Data preprocessing and feature engineering"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the dataset.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    df_processed = df.copy()
    
    # Drop 'id' column as it's not a feature
    if 'id' in df_processed.columns:
        df_processed = df_processed.drop('id', axis=1)
    
    # Create BMI feature: BMI = Weight (kg) / (Height (cm) / 100)²
    df_processed['BMI'] = df_processed['Peso'] / ((df_processed['Altura'] / 100) ** 2)
    
    logger.info("BMI feature created")
    
    return df_processed


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using one-hot encoding.
    
    Args:
        df: DataFrame with categorical features
        
    Returns:
        DataFrame with encoded features
    """
    df_encoded = pd.get_dummies(df, columns=['Cor do cabelo'], prefix='Hair')
    
    logger.info(f"Features encoded. New shape: {df_encoded.shape}")
    return df_encoded


def split_features_target(df: pd.DataFrame, target_column: str = 'Diabético') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split features and target variable.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of the target column
        
    Returns:
        Tuple of (features, target)
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y


class ScalerManager:
    """Manages scaling operations and persistence"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame) -> 'ScalerManager':
        """Fit the scaler on training data"""
        self.scaler.fit(X)
        self.is_fitted = True
        logger.info("Scaler fitted on training data")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transformation")
        
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        return X_scaled_df
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def save(self, filepath: str):
        """Save scaler to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {filepath}")
    
    def load(self, filepath: str):
        """Load scaler from disk"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_fitted = True
        logger.info(f"Scaler loaded from {filepath}")
