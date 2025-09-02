"""
Entropy Engine - A Python system for processing tokens through entropy-based transformation nodes.

This package provides a framework for creating complex token transformation pipelines
where tokens carry values and track their entropy, nodes apply transformations and can
branch dynamically, and entropy limits control when transformations stop.
"""

from .core import Token, EntropyNode, EntropyEngine

__version__ = "1.0.0"
__author__ = "Entropy Engine Contributors"
__email__ = "your.email@example.com"

__all__ = ["Token", "EntropyNode", "EntropyEngine"]