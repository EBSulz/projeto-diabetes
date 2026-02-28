"""Main training script"""
import sys
from pathlib import Path

# Add project root to path (go up one level from scripts/ to project root)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
from src.data.load_data import load_dataset, validate_data
from src.data.preprocessing import engineer_features, encode_categorical_features, split_features_target, ScalerManager
from src.models.train import (
    train_logistic_regression, train_random_forest, train_xgboost,
    train_svm, train_knn, log_model_mlflow
)
from src.utils.config import load_config, get_project_root
from src.utils.logging_config import setup_logging

# Set random seed
np.random.seed(42)


def main():
    """Main training pipeline"""
    # Setup
    project_root = get_project_root()
    config = load_config(project_root / "configs" / "config.yaml")
    
    setup_logging(
        log_level=config['logging']['level'],
        log_file=project_root / config['logging']['log_file']
    )
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Starting training pipeline")
    
    # Load data
    data_path = project_root / config['data']['raw_data_path']
    df = load_dataset(str(data_path))
    validate_data(df)
    
    # Feature engineering
    df_processed = engineer_features(df)
    df_encoded = encode_categorical_features(df_processed)
    
    # Split features and target
    X, y = split_features_target(df_encoded)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state'],
        stratify=y if config['training']['stratify'] else None
    )
    
    # Scale features
    scaler = ScalerManager()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = project_root / config['models']['scaler_path']
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    scaler.save(str(scaler_path))
    
    # Setup MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    experiment_name = config['mlflow']['experiment_name']
    try:
        mlflow.create_experiment(experiment_name)
        logger.info(f"Created new experiment: {experiment_name}")
    except:
        mlflow.set_experiment(experiment_name)
        logger.info(f"Using existing experiment: {experiment_name}")
    
    # Train models
    models_to_train = {
        'Logistic_Regression': (train_logistic_regression, config['models_config']['logistic_regression']),
        'Random_Forest': (train_random_forest, config['models_config']['random_forest']),
        'XGBoost': (train_xgboost, config['models_config']['xgboost']),
        'SVM': (train_svm, config['models_config']['svm']),
        'KNN': (train_knn, config['models_config']['knn'])
    }
    
    trained_models = {}
    
    for model_name, (train_func, params) in models_to_train.items():
        logger.info(f"Training {model_name}...")
        model, metrics_train, metrics_test = train_func(
            X_train_scaled, y_train,
            X_test_scaled, y_test,
            params
        )
        
        log_model_mlflow(model, model_name, params, metrics_train, metrics_test)
        trained_models[model_name] = model
    
    logger.info("Training pipeline completed successfully")
    logger.info("View results in MLflow UI: mlflow ui")


if __name__ == "__main__":
    main()
