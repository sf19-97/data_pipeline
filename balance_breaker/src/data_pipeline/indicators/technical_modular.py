"""
Modular Technical Indicators Implementation

This module implements individual technical indicators using the modular pattern.
Each indicator is a separate class that can be registered and used independently.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Set, ClassVar

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.data_pipeline.indicators.modular_base import ModularIndicator, register_indicator


@register_indicator
@implements("IndicatorCalculator")
class RSIIndicator(ModularIndicator):
    """
    Relative Strength Index (RSI) Indicator
    
    Parameters:
    -----------
    period : int
        RSI calculation period (default: 14)
    """
    
    indicator_name: ClassVar[str] = "rsi"
    indicator_category: ClassVar[str] = "technical"
    required_columns: ClassVar[Set[str]] = {"close"}
    
    def __init__(self, parameters=None):
        default_params = {
            'period': 14,
        }
        super().__init__(parameters or default_params)
    
    def calculate_indicator(self, df, **kwargs) -> Dict[str, Any]:
        """Calculate RSI for the given dataframe"""
        period = self.parameters.get('period', 14)
        price = df['close']
        
        # Calculate price changes
        delta = price.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate relative strength (avoid divide by zero)
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        # Return dictionary of calculated columns
        return {'RSI': rsi}


@register_indicator
@implements("IndicatorCalculator")
class MovingAverageIndicator(ModularIndicator):
    """
    Moving Average Indicator (both Simple and Exponential)
    
    Parameters:
    -----------
    ma_type : str
        Type of moving average ("sma" or "ema", default: "sma")
    periods : list
        List of periods to calculate (default: [10, 20, 50, 200])
    price_field : str
        Price field to use (default: "close")
    """
    
    indicator_name: ClassVar[str] = "ma"
    indicator_category: ClassVar[str] = "technical"
    required_columns: ClassVar[Set[str]] = {"close"}
    
    def __init__(self, parameters=None):
        default_params = {
            'ma_type': 'sma',
            'periods': [10, 20, 50, 200],
            'price_field': 'close'
        }
        super().__init__(parameters or default_params)
    
    def calculate_indicator(self, df, **kwargs) -> Dict[str, Any]:
        """Calculate Moving Averages for the given dataframe"""
        ma_type = self.parameters.get('ma_type', 'sma').lower()
        periods = self.parameters.get('periods', [10, 20, 50, 200])
        price_field = self.parameters.get('price_field', 'close')
        
        # Get price series to use
        price = df[price_field]
        
        # Calculate moving averages
        result = {}
        
        for period in periods:
            if ma_type == 'sma':
                # Simple Moving Average
                result[f'SMA_{period}'] = price.rolling(window=period).mean()
            elif ma_type == 'ema':
                # Exponential Moving Average
                result[f'EMA_{period}'] = price.ewm(span=period, adjust=False).mean()
            else:
                # Both types
                result[f'SMA_{period}'] = price.rolling(window=period).mean()
                result[f'EMA_{period}'] = price.ewm(span=period, adjust=False).mean()
        
        return result


@register_indicator
@implements("IndicatorCalculator")
class MACDIndicator(ModularIndicator):
    """
    Moving Average Convergence Divergence (MACD) Indicator
    
    Parameters:
    -----------
    fast_period : int
        Fast EMA period (default: 12)
    slow_period : int
        Slow EMA period (default: 26)
    signal_period : int
        Signal line period (default: 9)
    """
    
    indicator_name: ClassVar[str] = "macd"
    indicator_category: ClassVar[str] = "technical"
    required_columns: ClassVar[Set[str]] = {"close"}
    
    def __init__(self, parameters=None):
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }
        super().__init__(parameters or default_params)
    
    def calculate_indicator(self, df, **kwargs) -> Dict[str, Any]:
        """Calculate MACD for the given dataframe"""
        fast_period = self.parameters.get('fast_period', 12)
        slow_period = self.parameters.get('slow_period', 26)
        signal_period = self.parameters.get('signal_period', 9)
        
        # Get price series
        price = df['close']
        
        # Calculate fast and slow EMAs
        fast_ema = price.ewm(span=fast_period, adjust=False).mean()
        slow_ema = price.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Return dictionary of calculated columns
        return {
            'MACD': macd_line,
            'MACD_Signal': signal_line,
            'MACD_Hist': histogram
        }


@register_indicator
@implements("IndicatorCalculator")
class MomentumIndicator(ModularIndicator):
    """
    Momentum Indicator
    
    Parameters:
    -----------
    periods : list
        List of periods to calculate momentum (default: [5, 10, 20])
    """
    
    indicator_name: ClassVar[str] = "momentum"
    indicator_category: ClassVar[str] = "technical"
    required_columns: ClassVar[Set[str]] = {"close"}
    
    def __init__(self, parameters=None):
        default_params = {
            'periods': [5, 10, 20]
        }
        super().__init__(parameters or default_params)
    
    def calculate_indicator(self, df, **kwargs) -> Dict[str, Any]:
        """Calculate Momentum for the given dataframe"""
        periods = self.parameters.get('periods', [5, 10, 20])
        price = df['close']
        
        result = {}
        
        for period in periods:
            # Momentum = Current Price - Price N periods ago
            result[f'momentum_{period}'] = price - price.shift(period)
            
            # Rate of Change (percentage)
            result[f'roc_{period}'] = (price / price.shift(period) - 1) * 100
            
        return result


@register_indicator
@implements("IndicatorCalculator")
class VolatilityIndicator(ModularIndicator):
    """
    Volatility Indicator
    
    Parameters:
    -----------
    periods : list
        List of periods to calculate volatility (default: [5, 10, 20])
    """
    
    indicator_name: ClassVar[str] = "volatility"
    indicator_category: ClassVar[str] = "technical"
    required_columns: ClassVar[Set[str]] = {"close"}
    
    def __init__(self, parameters=None):
        default_params = {
            'periods': [5, 10, 20]
        }
        super().__init__(parameters or default_params)
    
    def calculate_indicator(self, df, **kwargs) -> Dict[str, Any]:
        """Calculate Volatility for the given dataframe"""
        periods = self.parameters.get('periods', [5, 10, 20])
        price = df['close']
        returns = price.pct_change()
        
        result = {}
        
        for period in periods:
            # Standard deviation of returns
            result[f'volatility_{period}'] = returns.rolling(window=period).std()
            
        return result