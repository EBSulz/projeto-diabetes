"""Data loading utilities"""
import pandas as pd
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Load the diabetes dataset from Excel file.
    
    Args:
        data_path: Path to the Excel file
        
    Returns:
        DataFrame with the loaded data
    """
    try:
        df = pd.read_excel(data_path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the loaded dataset.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises exception otherwise
    """
    required_columns = ['id', 'Cor do cabelo', 'Peso', 'Altura', 'Diabético']
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.isnull().sum().sum() > 0:
        logger.warning("Dataset contains missing values")
    
    logger.info("Data validation passed")
    return True
