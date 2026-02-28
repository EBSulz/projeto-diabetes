# Project Structure - Visual Guide

## 📁 Complete Directory Tree

```
projeto-diabetes/
│
├── 📂 src/                          # Source code package
│   ├── __init__.py                  # Package initialization
│   │
│   ├── 📂 data/                     # Data processing modules
│   │   ├── __init__.py
│   │   ├── load_data.py             # Data loading & validation
│   │   └── preprocessing.py         # Feature engineering & scaling
│   │
│   ├── 📂 models/                   # Model training & evaluation
│   │   ├── __init__.py
│   │   ├── train.py                 # Model training functions
│   │   └── evaluate.py              # Model evaluation utilities
│   │
│   └── 📂 utils/                    # Utility modules
│       ├── __init__.py
│       ├── config.py                # Configuration management
│       └── logging_config.py        # Logging setup
│
├── 📂 scripts/                      # Executable scripts
│   ├── __init__.py
│   ├── train.py                     # Main training pipeline
│   └── streamlit_app.py             # Streamlit dashboard
│
├── 📂 notebooks/                    # Jupyter notebooks
│   └── diabetes_prediction.ipynb   # Original notebook
│
├── 📂 tests/                        # Test suite
│   ├── __init__.py
│   ├── test_data.py                 # Data processing tests
│   └── test_models.py               # Model training tests
│
├── 📂 configs/                      # Configuration files
│   └── config.yaml                  # Main configuration
│
├── 📂 data/                         # Data directory
│   ├── 📂 raw/                      # Raw data files
│   │   ├── .gitkeep
│   │   └── diabetes_dataset.xlsx   # Dataset (gitignored)
│   └── 📂 processed/                # Processed data (generated)
│       └── .gitkeep
│
├── 📂 models/                        # Model artifacts
│   ├── .gitkeep
│   └── scaler.pkl                   # Saved scaler (generated)
│
├── 📂 docs/                         # Documentation
│   ├── README.md                    # Documentation index
│   ├── QUICKSTART.md                # Quick start guide
│   ├── PROJECT_STRUCTURE.md         # Structure details
│   └── PROJECT_STRUCTURE_VISUAL.md  # This file
│
├── 📂 logs/                         # Log files (generated)
│
├── 📂 mlruns/                       # MLflow tracking (generated)
│
├── 📂 .github/                      # GitHub configuration
│   └── 📂 workflows/
│       └── ci.yml                   # CI/CD pipeline
│
├── 📄 README.md                     # Main project README
├── 📄 CONTRIBUTING.md               # Contribution guidelines
├── 📄 CHANGELOG.md                  # Version history
├── 📄 ORGANIZATION_SUMMARY.md       # Organization summary
│
├── 📄 requirements.txt              # Base requirements
├── 📄 requirements-dev.txt          # Development requirements
├── 📄 requirements-prod.txt         # Production requirements
│
├── 📄 Makefile                      # Common tasks automation
├── 📄 setup.py                      # Package setup
├── 📄 pyproject.toml                # Modern Python config
├── 📄 pytest.ini                   # Pytest configuration
├── 📄 .flake8                       # Flake8 configuration
├── 📄 .pre-commit-config.yaml       # Pre-commit hooks
├── 📄 .gitignore                    # Git ignore rules
└── 📄 .env.template                 # Environment variables template
```

## 🎯 Directory Purposes

### `src/` - Source Code
- **Purpose**: Importable Python package
- **Structure**: Modular organization by functionality
- **Best Practice**: Clean separation of concerns

### `scripts/` - Executable Scripts
- **Purpose**: Entry points for running the application
- **Files**: Training script, Streamlit app
- **Best Practice**: Separate from source code

### `notebooks/` - Jupyter Notebooks
- **Purpose**: Exploratory data analysis and prototyping
- **Best Practice**: Keep root directory clean

### `tests/` - Test Suite
- **Purpose**: Unit tests and integration tests
- **Structure**: Mirrors `src/` structure
- **Best Practice**: Comprehensive coverage

### `configs/` - Configuration
- **Purpose**: YAML configuration files
- **Best Practice**: Centralized configuration management

### `data/` - Data Storage
- **Purpose**: Raw and processed data
- **Structure**: `raw/` for input, `processed/` for output
- **Best Practice**: Version control structure, not data

### `models/` - Model Artifacts
- **Purpose**: Saved models and scalers
- **Best Practice**: Gitignored, tracked via MLflow

### `docs/` - Documentation
- **Purpose**: All project documentation
- **Best Practice**: Organized and comprehensive

## 🔄 Data Flow

```
Raw Data (data/raw/)
    ↓
Data Loading (src/data/load_data.py)
    ↓
Preprocessing (src/data/preprocessing.py)
    ↓
Feature Engineering
    ↓
Model Training (src/models/train.py)
    ↓
MLflow Tracking (mlruns/)
    ↓
Model Artifacts (models/)
    ↓
Evaluation (src/models/evaluate.py)
    ↓
Streamlit Dashboard (scripts/streamlit_app.py)
```

## 📦 Package Structure

The `src/` package is organized as:

```
src/
├── data/          # Data layer
│   ├── Loading
│   └── Preprocessing
│
├── models/        # Model layer
│   ├── Training
│   └── Evaluation
│
└── utils/         # Utility layer
    ├── Config
    └── Logging
```

## 🛠️ Tool Integration

- **Black**: Code formatting (via pre-commit)
- **isort**: Import sorting (via pre-commit)
- **Flake8**: Linting (via pre-commit & CI)
- **mypy**: Type checking (via pre-commit & CI)
- **pytest**: Testing (via CI)
- **MLflow**: Experiment tracking
- **Streamlit**: Dashboard
- **GitHub Actions**: CI/CD

## ✅ Best Practices Checklist

- [x] Modular code structure
- [x] Separation of scripts and source
- [x] Comprehensive testing
- [x] Configuration management
- [x] Documentation organization
- [x] Pre-commit hooks
- [x] CI/CD pipeline
- [x] Type hints
- [x] Code formatting automation
- [x] Environment variable management
- [x] Proper .gitignore
- [x] Makefile for common tasks
- [x] Separate dev/prod requirements

## 🚀 Quick Navigation

- **Start here**: `README.md`
- **Quick setup**: `docs/QUICKSTART.md`
- **Structure details**: `docs/PROJECT_STRUCTURE.md`
- **Contribute**: `CONTRIBUTING.md`
- **Changes**: `CHANGELOG.md`
