"""
MatrixTransformer
================

A unified Python framework for structure-preserving matrix transformations in high-dimensional decision space.

This package provides tools for:
- Structure-preserving matrix transformations
- Quantifying information-structure trade-offs
- Interpolation between matrix types
- Custom matrix definitions
- Applications in ML, signal processing, quantum simulation, and more
"""

__version__ = '0.1.0'

# Fixed the typo in the filename
from .matrixtransformer import MatrixTransformer, MatrixType
from .base import *
from .base_classes import *
from .graph import *

# Main exports
__all__ = [
    'MatrixTransformer',
    'MatrixType',
]