"""
Signal Generators

This module provides generators for converting data into signals.
"""

from balance_breaker.src.signal.generators.ma_cross_generator import MovingAverageCrossSignalGenerator
from balance_breaker.src.signal.generators.pattern_generator import PatternRecognitionSignalGenerator

__all__ = [
    'MovingAverageCrossSignalGenerator',
    'PatternRecognitionSignalGenerator'
]