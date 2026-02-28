# Diabetes Prediction MLOps Project

A comprehensive MLOps project for predicting diabetes using machine learning models. This project follows industry best practices for machine learning operations, including modular code structure, experiment tracking with MLflow, and an interactive Streamlit dashboard.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Documentation](#documentation)

## ✨ Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, SVM, and KNN
- **MLflow Integration**: Automatic experiment tracking and model versioning
- **Streamlit Dashboard**: Interactive web application for data visualization and predictions
- **Modular Architecture**: Clean separation of concerns following MLOps best practices
- **CI/CD Pipeline**: Automated testing and code quality checks
- **Comprehensive Testing**: Unit tests for data processing and model training
- **Configuration Management**: YAML-based configuration for easy parameter tuning
- **Pre-commit Hooks**: Automated code quality checks before commits
- **Type Hints**: Full type annotation support

## 📁 Project Structure

```
projeto-diabetes/
├── src/                      # Source code package
│   ├── data/                 # Data loading and preprocessing
│   ├── models/               # Model training and evaluation
│   └── utils/                # Utility functions
├── scripts/                  # Executable scripts
│   ├── train.py             # Training pipeline
│   └── streamlit_app.py     # Streamlit dashboard
├── notebooks/                # Jupyter notebooks
├── tests/                    # Test suite
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── raw/                  # Raw data files
│   └── processed/            # Processed data
├── models/                   # Trained models and artifacts
├── docs/                     # Documentation
├── logs/                     # Log files
├── .github/                  # GitHub Actions workflows
│   └── workflows/
│       └── ci.yml
├── requirements.txt          # Base requirements
├── requirements-dev.txt     # Development requirements
├── requirements-prod.txt    # Production requirements
├── Makefile                 # Common tasks
├── setup.py                 # Package setup
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Create conda environment
conda create -n diabetes-ml python=3.9
conda activate diabetes-ml

# Install dependencies
make install-dev
# or
pip install -r requirements-dev.txt
```

### 2. Setup Pre-commit Hooks

```bash
make setup-precommit
# or
pre-commit install
```

### 3. Train Models

```bash
make train
# or
python scripts/train.py
```

### 4. Launch Dashboard

```bash
make streamlit
# or
streamlit run scripts/streamlit_app.py
```

## 📦 Installation

### Development Installation

```bash
make install-dev
```

This installs:
- Base requirements
- Testing tools (pytest, coverage)
- Code quality tools (black, flake8, mypy)
- Pre-commit hooks
- Jupyter notebooks

### Production Installation

```bash
make install-prod
```

This installs only production dependencies (no dev tools).

## 📖 Usage

### Training Models

```bash
make train
```

This will:
- Load and preprocess the data
- Train 5 different models
- Log all experiments to MLflow
- Save the scaler for inference

### Viewing MLflow UI

```bash
make mlflow-ui
# or
mlflow ui --port 5000
```

Open browser: `http://localhost:5000`

### Running Streamlit Dashboard

```bash
make streamlit
```

The dashboard provides:
- **Data Overview**: Dataset statistics and visualizations
- **Model Performance**: Best model metrics and hyperparameters
- **Prediction**: Interactive diabetes risk prediction
- **Model Comparison**: Side-by-side comparison of all models

## 🛠️ Development

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Type checking
make type-check

# Run all quality checks
make quality
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov
```

### Pre-commit Hooks

Pre-commit hooks automatically run on commit:
- Code formatting (Black)
- Import sorting (isort)
- Linting (Flake8)
- Type checking (mypy)
- Security checks (Bandit)

Run manually:
```bash
pre-commit run --all-files
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data.py -v
```

## 🔄 CI/CD

The project includes a GitHub Actions CI pipeline that runs:
- Code formatting checks
- Linting
- Unit tests with coverage
- Type checking

See `.github/workflows/ci.yml` for details.

## 📚 Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
- [Project Structure](docs/PROJECT_STRUCTURE.md)
- [Full README](docs/README.md)

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run quality checks: `make quality`
4. Run tests: `make test`
5. Commit (pre-commit hooks will run automatically)
6. Submit a pull request

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- MLflow for experiment tracking
- Streamlit for the dashboard framework
- scikit-learn and XGBoost for machine learning models
