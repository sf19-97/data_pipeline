"""
Moving Average Cross Signal Generator

This component generates signals based on moving average crossovers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.signal.base import BaseSignalGenerator
from balance_breaker.src.signal.signal_models import (
    Signal, RawSignal, InterpretedSignal, ActionSignal,
    SignalMetadata, Timeframe, SignalType, 
    SignalDirection, SignalStrength, SignalConfidence
)

@implements("SignalGenerator")
class MovingAverageCrossSignalGenerator(BaseSignalGenerator):
    """
    Generates signals based on moving average crossovers
    
    Parameters:
    -----------
    fast_period : int
        Period for the fast moving average (default: 10)
    slow_period : int
        Period for the slow moving average (default: 20)
    signal_type : str
        Type of signal to generate ('raw', 'interpreted', 'action') (default: 'interpreted')
    min_strength : int
        Minimum strength threshold (default: 1)
    lookback_periods : int
        Number of periods to look back for confirmation (default: 3)
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'fast_period': 10,
            'slow_period': 20,
            'signal_type': 'interpreted',
            'min_strength': 1,
            'lookback_periods': 3
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def generate_signals(self, data: Any, context: Dict[str, Any]) -> List[Signal]:
        """Generate moving average crossover signals
        
        Args:
            data: Price data dictionary by pair or DataFrame
            context: Generation context
            
        Returns:
            List of generated signals
        """
        signals = []
        
        # Handle different data formats
        if isinstance(data, dict) and 'price' in data:
            # Data pipeline format with 'price' key
            for pair, df in data['price'].items():
                pair_signals = self._generate_signals_for_pair(pair, df, context)
                signals.extend(pair_signals)
                
        elif isinstance(data, dict) and all(isinstance(df, pd.DataFrame) for df in data.values()):
            # Dictionary of DataFrames by pair
            for pair, df in data.items():
                pair_signals = self._generate_signals_for_pair(pair, df, context)
                signals.extend(pair_signals)
                
        elif isinstance(data, pd.DataFrame):
            # Single DataFrame
            pair = context.get('symbol', 'unknown')
            pair_signals = self._generate_signals_for_pair(pair, data, context)
            signals.extend(pair_signals)
        
        return signals
    
    def can_generate(self, data: Any, context: Dict[str, Any]) -> bool:
        """Check if generator can produce signals from the given data
        
        Args:
            data: Input data to check
            context: Generation context
            
        Returns:
            True if generator can produce signals, False otherwise
        """
        # Check data format
        if isinstance(data, dict) and 'price' in data:
            # Check if any price dataframe has enough data
            for pair, df in data['price'].items():
                if self._has_sufficient_data(df):
                    return True
            return False
            
        elif isinstance(data, dict) and all(isinstance(df, pd.DataFrame) for df in data.values()):
            # Check if any dataframe has enough data
            for pair, df in data.items():
                if self._has_sufficient_data(df):
                    return True
            return False
            
        elif isinstance(data, pd.DataFrame):
            # Check if dataframe has enough data
            return self._has_sufficient_data(data)
            
        return False
    
    @property
    def supported_timeframes(self) -> Set[Timeframe]:
        """Get supported timeframes for this generator"""
        return {
            Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.M30,
            Timeframe.H1, Timeframe.H4, Timeframe.D1, Timeframe.W1
        }
    
    @property
    def required_data_types(self) -> Set[str]:
        """Get required data types for this generator"""
        return {'price'}
    
    def _has_sufficient_data(self, df: pd.DataFrame) -> bool:
        """Check if dataframe has enough data for signal generation"""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return False
        
        # Check if required columns exist
        if 'close' not in df.columns:
            return False
        
        # Check if enough data points
        slow_period = self.parameters.get('slow_period', 20)
        required_points = slow_period + self.parameters.get('lookback_periods', 3)
        
        return len(df) >= required_points
    
    def _generate_signals_for_pair(self, pair: str, df: pd.DataFrame, 
                                 context: Dict[str, Any]) -> List[Signal]:
        """Generate signals for a single pair"""
        try:
            signals = []
            
            # Get parameters
            fast_period = self.parameters.get('fast_period', 10)
            slow_period = self.parameters.get('slow_period', 20)
            min_strength = self.parameters.get('min_strength', 1)
            lookback_periods = self.parameters.get('lookback_periods', 3)
            signal_type = self.parameters.get('signal_type', 'interpreted')
            
            # Check if enough data
            if not self._has_sufficient_data(df):
                return []
            
            # Calculate moving averages
            df = df.copy()
            if f'MA_{fast_period}' not in df.columns:
                df[f'MA_{fast_period}'] = df['close'].rolling(window=fast_period).mean()
            
            if f'MA_{slow_period}' not in df.columns:
                df[f'MA_{slow_period}'] = df['close'].rolling(window=slow_period).mean()
            
            # Calculate crossovers
            df['cross_above'] = (
                (df[f'MA_{fast_period}'] > df[f'MA_{slow_period}']) & 
                (df[f'MA_{fast_period}'].shift(1) <= df[f'MA_{slow_period}'].shift(1))
            )
            
            df['cross_below'] = (
                (df[f'MA_{fast_period}'] < df[f'MA_{slow_period}']) & 
                (df[f'MA_{fast_period}'].shift(1) >= df[f'MA_{slow_period}'].shift(1))
            )
            
            # Get the last row with a crossover
            last_cross_above = df['cross_above'].iloc[-lookback_periods:].any()
            last_cross_below = df['cross_below'].iloc[-lookback_periods:].any()
            
            if not (last_cross_above or last_cross_below):
                # No recent crossovers
                return []
            
            # Determine signal direction
            direction = SignalDirection.BULLISH if last_cross_above else SignalDirection.BEARISH
            
            # Determine signal strength based on price-MA distance
            last_price = df['close'].iloc[-1]
            last_fast_ma = df[f'MA_{fast_period}'].iloc[-1]
            last_slow_ma = df[f'MA_{slow_period}'].iloc[-1]
            
            if direction == SignalDirection.BULLISH:
                ma_diff = ((last_fast_ma / last_slow_ma) - 1) * 100
                strength_factor = min(4, max(1, int(ma_diff) + 1))
            else:
                ma_diff = ((last_slow_ma / last_fast_ma) - 1) * 100
                strength_factor = min(4, max(1, int(ma_diff) + 1))
            
            # Map strength factor to enum
            strength_map = {
                1: SignalStrength.WEAK,
                2: SignalStrength.MODERATE,
                3: SignalStrength.STRONG,
                4: SignalStrength.VERY_STRONG
            }
            strength = strength_map.get(strength_factor, SignalStrength.MODERATE)
            
            # Skip if strength below minimum
            if strength.value < min_strength:
                return []
            
            # Create signal metadata
            metadata = self.create_signal_metadata(context)
            
            # Get timeframe from context or default to H1
            timeframe_str = context.get('timeframe', 'H1')
            timeframe = next((tf for tf in Timeframe if tf.value == timeframe_str), Timeframe.H1)
            
            # Create signal based on signal type
            if signal_type == 'raw':
                signal = RawSignal(
                    symbol=pair,
                    timeframe=timeframe,
                    signal_type=SignalType.RAW,
                    direction=direction,
                    strength=strength,
                    metadata=metadata,
                    data={
                        'price': last_price,
                        'fast_ma': last_fast_ma,
                        'slow_ma': last_slow_ma,
                        'ma_diff': ma_diff,
                        'cross_type': 'above' if last_cross_above else 'below'
                    }
                )
                signals.append(signal)
                
            elif signal_type == 'interpreted':
                cross_type = 'above' if last_cross_above else 'below'
                interpretation = (
                    f"Moving average crossover ({fast_period} {cross_type} {slow_period}) "
                    f"indicates {'bullish' if direction == SignalDirection.BULLISH else 'bearish'} trend"
                )
                
                signal = InterpretedSignal(
                    symbol=pair,
                    timeframe=timeframe,
                    signal_type=SignalType.INTERPRETED,
                    direction=direction,
                    strength=strength,
                    metadata=metadata,
                    interpretation=interpretation,
                    indicators={
                        'price': last_price,
                        'fast_ma': last_fast_ma,
                        'slow_ma': last_slow_ma,
                        'ma_diff': ma_diff,
                        'cross_type': cross_type
                    }
                )
                signals.append(signal)
                
            elif signal_type == 'action':
                action_type = 'BUY' if direction == SignalDirection.BULLISH else 'SELL'
                
                signal = ActionSignal(
                    symbol=pair,
                    timeframe=timeframe,
                    signal_type=SignalType.ACTION,
                    direction=direction,
                    strength=strength,
                    metadata=metadata,
                    action_type=action_type,
                    action_parameters={
                        'price': last_price,
                        'stop_loss': last_slow_ma if direction == SignalDirection.BULLISH else last_fast_ma,
                        'take_profit': last_price * (1.05 if direction == SignalDirection.BULLISH else 0.95)
                    }
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'pair': pair},
                subsystem='signal',
                component='MovingAverageCrossSignalGenerator'
            )
            return []