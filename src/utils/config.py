"""Configuration management"""
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # Ensure we return a dict, not None
    if config_data is None:
        config_data = {}
    elif not isinstance(config_data, dict):
        raise ValueError(f"Configuration file must contain a dictionary, got {type(config_data)}")
    
    logger.info(f"Configuration loaded from {config_path}")
    return config_data


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    This function looks for the project root by finding a marker file/directory
    (like setup.py, pyproject.toml, or .git) starting from the current file's location.
    """
    current = Path(__file__).resolve()
    
    # Start from the current file and go up until we find project root markers
    for parent in [current] + list(current.parents):
        # Check for common project root markers
        if (parent / "setup.py").exists() or \
           (parent / "pyproject.toml").exists() or \
           (parent / ".git").exists() or \
           (parent / "src").exists() and (parent / "configs").exists():
            return parent
    
    # Fallback: go up 3 levels from src/utils/config.py
    return Path(__file__).parent.parent.parent
