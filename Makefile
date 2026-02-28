.PHONY: help install install-dev install-prod test lint format type-check clean train streamlit mlflow-ui setup-precommit

# Default target
help:
	@echo "Available commands:"
	@echo "  make install          - Install base requirements"
	@echo "  make install-dev     - Install development requirements"
	@echo "  make install-prod    - Install production requirements"
	@echo "  make test            - Run tests"
	@echo "  make test-cov        - Run tests with coverage"
	@echo "  make lint            - Run linters (flake8, pylint)"
	@echo "  make format          - Format code (black, isort)"
	@echo "  make type-check      - Run type checking (mypy)"
	@echo "  make quality         - Run all quality checks"
	@echo "  make clean           - Clean generated files"
	@echo "  make train           - Run training pipeline"
	@echo "  make streamlit       - Launch Streamlit app"
	@echo "  make mlflow-ui       - Launch MLflow UI"
	@echo "  make setup-precommit - Setup pre-commit hooks"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

install-prod:
	pip install -r requirements-prod.txt

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Code quality
lint:
	flake8 src/ scripts/ --max-line-length=100 --exclude=__pycache__,*.pyc
	pylint src/ --max-line-length=100 --disable=C0111

format:
	black src/ scripts/ --line-length=100
	isort src/ scripts/ --profile=black --line-length=100

type-check:
	mypy src/ --ignore-missing-imports --no-strict-optional

quality: lint type-check test

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -r {} + 2>/dev/null || true
	rm -f .coverage coverage.xml
	@echo "Cleanup complete"

# Training and serving
train:
	python scripts/train.py

streamlit:
	streamlit run scripts/streamlit_app.py

mlflow-ui:
	mlflow ui --port 5000 --host 127.0.0.1

# Setup
setup-precommit:
	pre-commit install
	@echo "Pre-commit hooks installed"
