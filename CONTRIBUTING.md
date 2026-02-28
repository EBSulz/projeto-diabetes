# Contributing Guide

Thank you for your interest in contributing to the Diabetes Prediction MLOps project!

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone <your-fork-url>
   cd projeto-diabetes
   ```

2. **Create a conda environment**
   ```bash
   conda create -n diabetes-ml python=3.9
   conda activate diabetes-ml
   ```

3. **Install development dependencies**
   ```bash
   make install-dev
   # or
   pip install -r requirements-dev.txt
   ```

4. **Setup pre-commit hooks**
   ```bash
   make setup-precommit
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Follow PEP 8 style guide
   - Add type hints where appropriate
   - Write tests for new functionality

3. **Run quality checks**
   ```bash
   make quality
   ```

4. **Run tests**
   ```bash
   make test
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Description of changes"
   ```
   Pre-commit hooks will run automatically.

6. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- **Formatting**: Use Black (line length: 100)
- **Imports**: Use isort (Black profile)
- **Linting**: Follow Flake8 rules
- **Type Hints**: Add type hints for function signatures
- **Documentation**: Add docstrings to all functions and classes

## Testing Guidelines

- Write tests for all new functionality
- Aim for >80% code coverage
- Use descriptive test names
- Follow Arrange-Act-Assert pattern
- Use fixtures for common test data

## Pull Request Process

1. Ensure all tests pass
2. Ensure code quality checks pass
3. Update documentation if needed
4. Add a clear description of changes
5. Reference any related issues

## Code Review

- Be respectful and constructive
- Focus on code quality and best practices
- Ask questions if something is unclear
- Suggest improvements when appropriate

## Questions?

Feel free to open an issue for questions or discussions.
