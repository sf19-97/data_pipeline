"""
Pattern Recognition Signal Generator

This component generates signals based on candlestick pattern detection.
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
class PatternRecognitionSignalGenerator(BaseSignalGenerator):
    """
    Generates signals based on price pattern recognition
    
    Parameters:
    -----------
    patterns : List[str]
        List of patterns to detect (default: ['doji', 'engulfing', 'hammer', 'shooting_star'])
    min_confidence : int
        Minimum confidence level (default: 2)
    signal_type : str
        Type of signal to generate ('raw', 'interpreted', 'action') (default: 'interpreted')
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'patterns': ['doji', 'engulfing', 'hammer', 'shooting_star'],
            'min_confidence': 2,
            'signal_type': 'interpreted'
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def generate_signals(self, data: Any, context: Dict[str, Any]) -> List[Signal]:
        """Generate pattern recognition signals
        
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
            Timeframe.M15, Timeframe.M30, Timeframe.H1, 
            Timeframe.H4, Timeframe.D1, Timeframe.W1
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
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Need at least 5 rows for pattern detection
        return len(df) >= 5
    
    def _generate_signals_for_pair(self, pair: str, df: pd.DataFrame, 
                                 context: Dict[str, Any]) -> List[Signal]:
        """Generate signals for a single pair"""
        try:
            signals = []
            
            # Get parameters
            patterns = self.parameters.get('patterns', ['doji', 'engulfing', 'hammer', 'shooting_star'])
            min_confidence = self.parameters.get('min_confidence', 2)
            signal_type = self.parameters.get('signal_type', 'interpreted')
            
            # Check if enough data
            if not self._has_sufficient_data(df):
                return []
            
            # Detect patterns
            detected_patterns = []
            
            # Doji pattern
            if 'doji' in patterns:
                doji = self._detect_doji(df)
                if doji:
                    detected_patterns.append(doji)
            
            # Engulfing pattern
            if 'engulfing' in patterns:
                engulfing = self._detect_engulfing(df)
                if engulfing:
                    detected_patterns.append(engulfing)
            
            # Hammer pattern
            if 'hammer' in patterns:
                hammer = self._detect_hammer(df)
                if hammer:
                    detected_patterns.append(hammer)
            
            # Shooting star pattern
            if 'shooting_star' in patterns:
                shooting_star = self._detect_shooting_star(df)
                if shooting_star:
                    detected_patterns.append(shooting_star)
            
            # No patterns detected
            if not detected_patterns:
                return []
            
            # Get timeframe from context or default to H1
            timeframe_str = context.get('timeframe', 'H1')
            timeframe = next((tf for tf in Timeframe if tf.value == timeframe_str), Timeframe.H1)
            
            # Create signals for each pattern with confidence >= min_confidence
            for pattern in detected_patterns:
                if pattern['confidence'].value < min_confidence:
                    continue
                
                # Create signal metadata
                metadata = self.create_signal_metadata(context)
                metadata.confidence = pattern['confidence']
                
                if 'tags' not in metadata.tags:
                    metadata.tags.append('pattern')
                
                if pattern['pattern'] not in metadata.tags:
                    metadata.tags.append(pattern['pattern'])
                
                # Create signal based on signal type
                if signal_type == 'raw':
                    signal = RawSignal(
                        symbol=pair,
                        timeframe=timeframe,
                        signal_type=SignalType.RAW,
                        direction=pattern['direction'],
                        strength=pattern['strength'],
                        metadata=metadata,
                        data={
                            'pattern': pattern['pattern'],
                            'description': pattern['description'],
                            'price': df['close'].iloc[-1],
                            'pattern_data': pattern['data']
                        }
                    )
                    signals.append(signal)
                    
                elif signal_type == 'interpreted':
                    signal = InterpretedSignal(
                        symbol=pair,
                        timeframe=timeframe,
                        signal_type=SignalType.INTERPRETED,
                        direction=pattern['direction'],
                        strength=pattern['strength'],
                        metadata=metadata,
                        interpretation=pattern['description'],
                        indicators={
                            'pattern': pattern['pattern'],
                            'price': df['close'].iloc[-1],
                            'pattern_data': pattern['data']
                        }
                    )
                    signals.append(signal)
                    
                elif signal_type == 'action':
                    action_type = 'BUY' if pattern['direction'] == SignalDirection.BULLISH else 'SELL'
                    
                    # Calculate stop loss and take profit based on pattern
                    last_price = df['close'].iloc[-1]
                    atr = self._calculate_atr(df)
                    
                    if pattern['direction'] == SignalDirection.BULLISH:
                        stop_loss = last_price - (atr * 2)
                        take_profit = last_price + (atr * 3)
                    else:
                        stop_loss = last_price + (atr * 2)
                        take_profit = last_price - (atr * 3)
                    
                    signal = ActionSignal(
                        symbol=pair,
                        timeframe=timeframe,
                        signal_type=SignalType.ACTION,
                        direction=pattern['direction'],
                        strength=pattern['strength'],
                        metadata=metadata,
                        action_type=action_type,
                        action_parameters={
                            'price': last_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'pattern': pattern['pattern']
                        }
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'pair': pair},
                subsystem='signal',
                component='PatternRecognitionSignalGenerator'
            )
            return []
    
    def _detect_doji(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect doji candlestick pattern"""
        try:
            # Get the last candle
            last_candle = df.iloc[-1]
            
            # Calculate body size as percentage of total range
            body_size = abs(last_candle['close'] - last_candle['open'])
            total_range = last_candle['high'] - last_candle['low']
            
            if total_range == 0:
                return None
                
            body_ratio = body_size / total_range
            
            # Doji has a very small body (less than 10% of total range)
            if body_ratio < 0.1:
                # Check previous trend
                prev_candles = df.iloc[-6:-1]
                
                # Determine if in an uptrend or downtrend
                uptrend = prev_candles['close'].iloc[-1] > prev_candles['close'].iloc[0]
                
                # Direction depends on trend (reversal signal)
                direction = SignalDirection.BEARISH if uptrend else SignalDirection.BULLISH
                
                # Determine confidence based on position and previous trend strength
                trend_strength = abs(prev_candles['close'].iloc[-1] - prev_candles['close'].iloc[0])
                trend_strength_ratio = trend_strength / prev_candles['close'].iloc[0]
                
                confidence = SignalConfidence.PROBABLE
                if trend_strength_ratio > 0.03:  # Strong trend
                    confidence = SignalConfidence.LIKELY
                
                # Check if doji is at a significant level
                significance = self._check_level_significance(df)
                if significance > 0:
                    confidence = SignalConfidence.CONFIRMED
                
                return {
                    'pattern': 'doji',
                    'direction': direction,
                    'strength': SignalStrength.MODERATE,
                    'confidence': confidence,
                    'description': f"Doji pattern detected, signaling potential {'bearish' if direction == SignalDirection.BEARISH else 'bullish'} reversal",
                    'data': {
                        'body_ratio': body_ratio,
                        'upper_shadow': (last_candle['high'] - max(last_candle['open'], last_candle['close'])) / total_range,
                        'lower_shadow': (min(last_candle['open'], last_candle['close']) - last_candle['low']) / total_range
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting doji: {str(e)}")
            return None
    
    def _detect_engulfing(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect engulfing candlestick pattern"""
        try:
            if len(df) < 2:
                return None
                
            # Get the last two candles
            current_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            
            # Calculate bodies
            current_body_low = min(current_candle['open'], current_candle['close'])
            current_body_high = max(current_candle['open'], current_candle['close'])
            prev_body_low = min(prev_candle['open'], prev_candle['close'])
            prev_body_high = max(prev_candle['open'], prev_candle['close'])
            
            # Check if current candle engulfs previous candle's body
            if current_body_low <= prev_body_low and current_body_high >= prev_body_high:
                # Determine direction
                bullish = current_candle['close'] > current_candle['open']
                direction = SignalDirection.BULLISH if bullish else SignalDirection.BEARISH
                
                # Previous trend should be opposite of current candle direction
                prev_trend_compatible = (prev_candle['close'] < prev_candle['open']) if bullish else (prev_candle['close'] > prev_candle['open'])
                
                if not prev_trend_compatible:
                    return None
                
                # Determine strength by how much the current body engulfs the previous
                engulfing_ratio = (current_body_high - current_body_low) / (prev_body_high - prev_body_low)
                
                strength = SignalStrength.MODERATE
                if engulfing_ratio > 1.5:
                    strength = SignalStrength.STRONG
                if engulfing_ratio > 2.0:
                    strength = SignalStrength.VERY_STRONG
                
                # Determine confidence
                confidence = SignalConfidence.PROBABLE
                
                # Check if in a trend
                prev_candles = df.iloc[-6:-2]
                if len(prev_candles) >= 4:
                    trend_direction = 1 if prev_candles['close'].iloc[-1] > prev_candles['close'].iloc[0] else -1
                    pattern_agrees_with_trend = (direction == SignalDirection.BULLISH and trend_direction < 0) or (direction == SignalDirection.BEARISH and trend_direction > 0)
                    
                    if pattern_agrees_with_trend:
                        confidence = SignalConfidence.LIKELY
                
                # Check if at a significant level
                significance = self._check_level_significance(df)
                if significance > 0:
                    confidence = SignalConfidence.CONFIRMED
                
                return {
                    'pattern': 'engulfing',
                    'direction': direction,
                    'strength': strength,
                    'confidence': confidence,
                    'description': f"{'Bullish' if direction == SignalDirection.BULLISH else 'Bearish'} engulfing pattern detected",
                    'data': {
                        'engulfing_ratio': engulfing_ratio,
                        'current_body': current_body_high - current_body_low,
                        'prev_body': prev_body_high - prev_body_low
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting engulfing: {str(e)}")
            return None
    
    def _detect_hammer(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect hammer candlestick pattern"""
        try:
            # Get the last candle
            last_candle = df.iloc[-1]
            
            # Calculate body size as percentage of total range
            body_size = abs(last_candle['close'] - last_candle['open'])
            total_range = last_candle['high'] - last_candle['low']
            
            if total_range == 0:
                return None
                
            body_ratio = body_size / total_range
            
            # Hammer has a small body at the top and a long lower shadow
            body_at_top = min(last_candle['open'], last_candle['close']) - last_candle['low'] > 2 * body_size
            small_upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close']) < 0.2 * body_size
            
            if body_ratio < 0.3 and body_at_top and small_upper_shadow:
                # Check previous trend
                prev_candles = df.iloc[-6:-1]
                
                # Hammer is bullish if found in a downtrend
                if len(prev_candles) >= 3 and prev_candles['close'].iloc[-1] < prev_candles['close'].iloc[0]:
                    # Calculate shadow ratio
                    lower_shadow = min(last_candle['open'], last_candle['close']) - last_candle['low']
                    lower_shadow_ratio = lower_shadow / total_range
                    
                    # Determine strength based on shadow ratio
                    strength = SignalStrength.MODERATE
                    if lower_shadow_ratio > 0.66:
                        strength = SignalStrength.STRONG
                    
                    # Determine confidence
                    confidence = SignalConfidence.PROBABLE
                    
                    # Check significance of the level
                    significance = self._check_level_significance(df)
                    if significance > 0:
                        confidence = SignalConfidence.LIKELY
                        
                    return {
                        'pattern': 'hammer',
                        'direction': SignalDirection.BULLISH,
                        'strength': strength,
                        'confidence': confidence,
                        'description': "Hammer pattern detected, signaling potential bullish reversal",
                        'data': {
                            'body_ratio': body_ratio,
                            'lower_shadow_ratio': lower_shadow_ratio
                        }
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting hammer: {str(e)}")
            return None
    
    def _detect_shooting_star(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect shooting star candlestick pattern"""
        try:
            # Get the last candle
            last_candle = df.iloc[-1]
            
            # Calculate body size as percentage of total range
            body_size = abs(last_candle['close'] - last_candle['open'])
            total_range = last_candle['high'] - last_candle['low']
            
            if total_range == 0:
                return None
                
            body_ratio = body_size / total_range
            
            # Shooting star has a small body at the bottom and a long upper shadow
            body_at_bottom = last_candle['high'] - max(last_candle['open'], last_candle['close']) > 2 * body_size
            small_lower_shadow = min(last_candle['open'], last_candle['close']) - last_candle['low'] < 0.2 * body_size
            
            if body_ratio < 0.3 and body_at_bottom and small_lower_shadow:
                # Check previous trend
                prev_candles = df.iloc[-6:-1]
                
                # Shooting star is bearish if found in an uptrend
                if len(prev_candles) >= 3 and prev_candles['close'].iloc[-1] > prev_candles['close'].iloc[0]:
                    # Calculate shadow ratio
                    upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])
                    upper_shadow_ratio = upper_shadow / total_range
                    
                    # Determine strength based on shadow ratio
                    strength = SignalStrength.MODERATE
                    if upper_shadow_ratio > 0.66:
                        strength = SignalStrength.STRONG
                    
                    # Determine confidence
                    confidence = SignalConfidence.PROBABLE
                    
                    # Check significance of the level
                    significance = self._check_level_significance(df)
                    if significance > 0:
                        confidence = SignalConfidence.LIKELY
                        
                    return {
                        'pattern': 'shooting_star',
                        'direction': SignalDirection.BEARISH,
                        'strength': strength,
                        'confidence': confidence,
                        'description': "Shooting star pattern detected, signaling potential bearish reversal",
                        'data': {
                            'body_ratio': body_ratio,
                            'upper_shadow_ratio': upper_shadow_ratio
                        }
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting shooting star: {str(e)}")
            return None
    
    def _check_level_significance(self, df: pd.DataFrame) -> int:
        """Check if price is at a significant level (support/resistance)"""
        try:
            # Simple check for significant levels
            # More sophisticated algorithms could be implemented
            last_price = df['close'].iloc[-1]
            
            # Look back up to 20 bars
            lookback = min(20, len(df) - 1)
            price_history = df.iloc[-lookback-1:-1]
            
            # Check how many times price has reversed near current level
            price_range = price_history['high'].max() - price_history['low'].min()
            if price_range == 0:
                return 0
                
            significance_threshold = 0.01  # 1% of price range
            
            # Count reversal points near current price
            reversal_count = 0
            for i in range(1, len(price_history) - 1):
                # Local high
                if (price_history['high'].iloc[i] > price_history['high'].iloc[i-1] and 
                    price_history['high'].iloc[i] > price_history['high'].iloc[i+1]):
                    
                    if abs(price_history['high'].iloc[i] - last_price) / price_range < significance_threshold:
                        reversal_count += 1
                
                # Local low
                if (price_history['low'].iloc[i] < price_history['low'].iloc[i-1] and 
                    price_history['low'].iloc[i] < price_history['low'].iloc[i+1]):
                    
                    if abs(price_history['low'].iloc[i] - last_price) / price_range < significance_threshold:
                        reversal_count += 1
            
            return reversal_count
            
        except Exception as e:
            self.logger.error(f"Error checking level significance: {str(e)}")
            return 0
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range (ATR)"""
        try:
            if len(df) < period + 1:
                return df['high'].iloc[-1] - df['low'].iloc[-1]
            
            # Calculate True Range
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR as simple moving average of TR
            atr = tr.rolling(window=period).mean().iloc[-1]
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return df['high'].iloc[-1] - df['low'].iloc[-1]