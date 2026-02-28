# Quick Start Guide

This guide will help you get started with the Diabetes Prediction MLOps project quickly.

## Prerequisites

- Python 3.8+
- Conda (recommended) or pip
- Git (for version control)

## Step 1: Environment Setup

```bash
# Create conda environment
conda create -n diabetes-ml python=3.9
conda activate diabetes-ml

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Data Setup

Ensure your dataset is in the correct location:

```bash
# The dataset should be at:
data/raw/diabetes_dataset.xlsx
```

If you need to move it:
```bash
# Windows PowerShell
Copy-Item diabetes_dataset.xlsx -Destination data\raw\diabetes_dataset.xlsx

# Linux/Mac
cp diabetes_dataset.xlsx data/raw/diabetes_dataset.xlsx
```

## Step 3: Train Models

Run the training script:

```bash
python train.py
```

This will:
- Load and preprocess the data
- Train 5 ML models
- Log experiments to MLflow
- Save the scaler for inference

## Step 4: View Results

### Option 1: MLflow UI

```bash
mlflow ui --port 5000
```

Open browser: http://localhost:5000

### Option 2: Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard will open automatically in your browser.

## Step 5: Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

## Common Commands

```bash
# Format code
black src/ train.py streamlit_app.py
isort src/ train.py streamlit_app.py

# Lint code
flake8 src/ train.py streamlit_app.py

# Type check
mypy src/ --ignore-missing-imports
```

## Troubleshooting

### Issue: Module not found
**Solution**: Make sure you're in the project root directory and the conda environment is activated.

### Issue: Data file not found
**Solution**: Check that `data/raw/diabetes_dataset.xlsx` exists. If not, copy it from the project root.

### Issue: MLflow UI not starting
**Solution**: 
- Check if port 5000 is already in use
- Try a different port: `mlflow ui --port 5001`
- Ensure MLflow is installed: `pip install mlflow`

### Issue: Streamlit not working
**Solution**:
- Ensure Streamlit is installed: `pip install streamlit plotly`
- Check that models have been trained first (run `train.py`)

## Next Steps

1. Explore the Streamlit dashboard
2. Review model performance in MLflow UI
3. Experiment with different hyperparameters in `configs/config.yaml`
4. Add new features or models to the pipeline
