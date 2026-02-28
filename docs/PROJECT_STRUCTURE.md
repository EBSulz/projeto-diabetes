# Project Structure Documentation

This document describes the MLOps project structure and the purpose of each component.

## Directory Structure

```
projeto-diabetes/
│
├── src/                          # Source code package
│   ├── __init__.py               # Package initialization
│   │
│   ├── data/                     # Data processing modules
│   │   ├── __init__.py
│   │   ├── load_data.py          # Data loading and validation
│   │   └── preprocessing.py       # Feature engineering and scaling
│   │
│   ├── models/                   # Model training and evaluation
│   │   ├── __init__.py
│   │   ├── train.py              # Model training functions
│   │   └── evaluate.py           # Model evaluation utilities
│   │
│   └── utils/                    # Utility modules
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       └── logging_config.py     # Logging setup
│
├── configs/                      # Configuration files
│   └── config.yaml               # Main configuration file
│
├── data/                         # Data directory
│   ├── raw/                      # Raw data files
│   │   └── diabetes_dataset.xlsx
│   └── processed/                # Processed data (generated)
│
├── models/                       # Model artifacts
│   └── scaler.pkl                # Saved scaler (generated)
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_data.py              # Data processing tests
│   └── test_models.py             # Model training tests
│
├── logs/                         # Log files (generated)
│
├── mlruns/                       # MLflow tracking data (generated)
│
├── .github/                      # GitHub configuration
│   └── workflows/
│       └── ci.yml                # CI/CD pipeline
│
├── train.py                      # Main training script
├── streamlit_app.py              # Streamlit dashboard
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── pytest.ini                    # Pytest configuration
├── .flake8                       # Flake8 linting configuration
├── .gitignore                    # Git ignore rules
├── README.md                     # Main documentation
├── QUICKSTART.md                 # Quick start guide
└── PROJECT_STRUCTURE.md          # This file
```

## Module Descriptions

### Data Modules (`src/data/`)

#### `load_data.py`
- **Purpose**: Load and validate datasets
- **Key Functions**:
  - `load_dataset()`: Load Excel files
  - `validate_data()`: Validate data structure and completeness

#### `preprocessing.py`
- **Purpose**: Feature engineering and data preprocessing
- **Key Functions**:
  - `engineer_features()`: Create BMI feature
  - `encode_categorical_features()`: One-hot encode categorical variables
  - `split_features_target()`: Separate features and target
  - `ScalerManager`: Manage feature scaling with persistence

### Model Modules (`src/models/`)

#### `train.py`
- **Purpose**: Model training and MLflow logging
- **Key Functions**:
  - `calculate_metrics()`: Compute classification metrics
  - `log_model_mlflow()`: Log models to MLflow
  - `train_*()`: Training functions for each model type

#### `evaluate.py`
- **Purpose**: Model evaluation and comparison
- **Key Functions**:
  - `get_best_model()`: Retrieve best model from MLflow
  - `get_model_comparison()`: Compare all models
  - `evaluate_model_predictions()`: Detailed evaluation

### Utility Modules (`src/utils/`)

#### `config.py`
- **Purpose**: Configuration management
- **Key Functions**:
  - `load_config()`: Load YAML configuration
  - `get_project_root()`: Get project root directory

#### `logging_config.py`
- **Purpose**: Logging setup
- **Key Functions**:
  - `setup_logging()`: Configure logging

## Configuration Files

### `configs/config.yaml`
Central configuration file containing:
- Data paths
- Model hyperparameters
- MLflow settings
- Training parameters
- Logging configuration

## Scripts

### `train.py`
Main training pipeline that:
1. Loads configuration
2. Loads and preprocesses data
3. Trains multiple models
4. Logs experiments to MLflow
5. Saves artifacts

### `streamlit_app.py`
Interactive web application with:
- Data visualization
- Model performance metrics
- Interactive predictions
- Model comparison

## Testing

### Test Structure
- `tests/test_data.py`: Tests for data loading and preprocessing
- `tests/test_models.py`: Tests for model training

### Running Tests
```bash
pytest                    # Run all tests
pytest --cov=src          # With coverage
pytest tests/test_data.py # Specific test file
```

## CI/CD Pipeline

### `.github/workflows/ci.yml`
Automated pipeline that runs:
1. **Linting**: Code formatting and style checks
2. **Testing**: Unit tests with coverage
3. **Type Checking**: Static type analysis

## Best Practices Implemented

1. **Separation of Concerns**: Clear module boundaries
2. **Configuration Management**: Centralized YAML config
3. **Logging**: Structured logging throughout
4. **Testing**: Comprehensive unit tests
5. **Version Control**: Proper .gitignore for artifacts
6. **Documentation**: README, quick start, and structure docs
7. **CI/CD**: Automated quality checks
8. **Experiment Tracking**: MLflow integration
9. **Reproducibility**: Fixed random seeds, versioned models
10. **Modularity**: Reusable functions and classes

## Adding New Features

### Adding a New Model
1. Add training function to `src/models/train.py`
2. Add hyperparameters to `configs/config.yaml`
3. Add to training loop in `train.py`
4. Add tests to `tests/test_models.py`

### Adding New Features
1. Add feature engineering to `src/data/preprocessing.py`
2. Update configuration if needed
3. Update tests
4. Update documentation

### Adding New Visualizations
1. Add visualization functions to `streamlit_app.py`
2. Create new page or section
3. Update navigation if needed
