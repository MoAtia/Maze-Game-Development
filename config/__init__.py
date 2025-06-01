"""Configuration package for hand gesture classification project.

This package contains:
- paths.py: Path configurations
- params.py: Hyperparameters and constants
"""

# Import specific items from submodules
from .pathes import BASE_DIR, DATA_DIR, MODELS_DIR
from .params import MODELS, DATA_PATH, TEST_SIZE, RANDOM_STATE, SCALER, ENCODER

# Explicitly list what should be available when importing from package
__all__ = [
    'BASE_DIR',
    'DATA_DIR',
    'MODELS_DIR',
    'MODELS',
    'DATA_PATH',
    'TEST_SIZE',
    'RANDOM_STATE',
    'SCALER',
    'ENCODER'
]