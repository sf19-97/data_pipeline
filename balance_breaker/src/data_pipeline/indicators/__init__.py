"""
Indicators - Components for calculating financial and economic indicators

This module provides calculators for technical, economic, and composite indicators.
"""

from .technical import TechnicalIndicators
from .economic import EconomicIndicators
from .composite import CompositeIndicators

# Add import for modular indicator implementations
try:
    from .technical_modular import *
except ImportError:
    pass  # Modular indicators not available

__all__ = [
    'TechnicalIndicators',
    'EconomicIndicators',
    'CompositeIndicators'
]