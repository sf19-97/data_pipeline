"""
Correlation Signal Combiner

This component combines signals based on correlation between signals.
"""

import copy
from typing import Dict, List, Any, Optional, Set

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.signal.base import BaseSignalCombiner
from balance_breaker.src.signal.signal_models import (
    Signal, InterpretedSignal, SignalMetadata, Timeframe,
    SignalType, SignalDirection, SignalStrength, SignalConfidence
)

@implements("SignalCombiner")
class CorrelationSignalCombiner(BaseSignalCombiner):
    """
    Combines signals based on correlation between signals
    
    Parameters:
    -----------
    min_signals : int
        Minimum number of signals required to generate a combined signal (default: 2)
    correlation_threshold : float
        Minimum correlation threshold (default: 0.7)
    combination_method : str
        Method for combining signals ('avg', 'max', 'weighted') (default: 'weighted')
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'min_signals': 2,
            'correlation_threshold': 0.7,
            'combination_method': 'weighted'
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def combine_signals(self, signals: List[Signal], context: Dict[str, Any]) -> List[Signal]:
        """Combine signals into new signals
        
        Args:
            signals: Input signals to combine
            context: Combination context
            
        Returns:
            Combined signals
        """
        try:
            # Get parameters
            min_signals = self.parameters.get('min_signals', 2)
            combination_method = self.parameters.get('combination_method', 'weighted')
            
            if len(signals) < min_signals:
                return []
            
            # Group signals by symbol
            symbol_groups = {}
            for signal in signals:
                if signal.symbol not in symbol_groups:
                    symbol_groups[signal.symbol] = []
                symbol_groups[signal.symbol].append(signal)
            
            combined_signals = []
            
            # Process each symbol group
            for symbol, symbol_signals in symbol_groups.items():
                if len(symbol_signals) < min_signals:
                    continue
                
                # Group signals by timeframe
                timeframe_groups = {}
                for signal in symbol_signals:
                    tf = signal.timeframe.value
                    if tf not in timeframe_groups:
                        timeframe_groups[tf] = []
                    timeframe_groups[tf].append(signal)
                
                # Process each timeframe group
                for timeframe, tf_signals in timeframe_groups.items():
                    if len(tf_signals) < min_signals:
                        continue
                    
                    # Combine signals
                    combined_signal = self._combine_signals_by_method(
                        tf_signals, 
                        symbol, 
                        timeframe, 
                        combination_method,
                        context
                    )
                    
                    if combined_signal:
                        combined_signals.append(combined_signal)
            
            return combined_signals
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'signal_count': len(signals)},
                subsystem='signal',
                component='CorrelationSignalCombiner'
            )
            return []
    
    def can_combine(self, signals: List[Signal]) -> bool:
        """Check if combiner can work with the given signals
        
        Args:
            signals: Signals to check
            
        Returns:
            True if signals can be combined, False otherwise
        """
        # Need at least min_signals signals
        min_signals = self.parameters.get('min_signals', 2)
        if len(signals) < min_signals:
            return False
        
        # Check if there are enough signals for any symbol
        symbol_counts = {}
        for signal in signals:
            symbol_counts[signal.symbol] = symbol_counts.get(signal.symbol, 0) + 1
        
        return any(count >= min_signals for count in symbol_counts.values())
    
    def _combine_signals_by_method(self, signals: List[Signal], 
                                 symbol: str, 
                                 timeframe: str,
                                 method: str,
                                 context: Dict[str, Any]) -> Optional[Signal]:
        """Combine signals using the specified method"""
        if method == 'avg':
            return self._combine_signals_avg(signals, symbol, timeframe, context)
        elif method == 'max':
            return self._combine_signals_max(signals, symbol, timeframe, context)
        elif method == 'weighted':
            return self._combine_signals_weighted(signals, symbol, timeframe, context)
        else:
            self.logger.warning(f"Unknown combination method: {method}")
            return None
    
    def _combine_signals_avg(self, signals: List[Signal], 
                          symbol: str, 
                          timeframe: str,
                          context: Dict[str, Any]) -> Optional[Signal]:
        """Combine signals using average method"""
        # Count signals by direction
        direction_counts = {
            SignalDirection.BULLISH: 0,
            SignalDirection.BEARISH: 0,
            SignalDirection.NEUTRAL: 0
        }
        
        for signal in signals:
            direction_counts[signal.direction] += 1
        
        # Determine majority direction
        max_direction = max(direction_counts.items(), key=lambda x: x[1])[0]
        
        # If tied, return neutral
        if direction_counts[SignalDirection.BULLISH] == direction_counts[SignalDirection.BEARISH]:
            max_direction = SignalDirection.NEUTRAL
        
        # Calculate average strength
        directional_signals = [s for s in signals if s.direction == max_direction]
        if not directional_signals:
            return None
            
        avg_strength_value = sum(s.strength.value for s in directional_signals) / len(directional_signals)
        
        # Map to nearest strength enum
        strength_map = {
            1: SignalStrength.WEAK,
            2: SignalStrength.MODERATE,
            3: SignalStrength.STRONG,
            4: SignalStrength.VERY_STRONG
        }
        
        # Round to nearest integer
        strength_value = round(avg_strength_value)
        strength = strength_map.get(strength_value, SignalStrength.MODERATE)
        
        # Calculate average confidence
        avg_confidence_value = sum(s.metadata.confidence.value for s in directional_signals) / len(directional_signals)
        
        # Map to nearest confidence enum
        confidence_map = {
            1: SignalConfidence.SPECULATIVE,
            2: SignalConfidence.PROBABLE,
            3: SignalConfidence.LIKELY,
            4: SignalConfidence.CONFIRMED
        }
        
        # Round to nearest integer
        confidence_value = round(avg_confidence_value)
        confidence = confidence_map.get(confidence_value, SignalConfidence.PROBABLE)
        
        # Create metadata
        sources = [s.metadata.source for s in directional_signals]
        metadata = SignalMetadata(
            source=f"Combined({','.join(set(sources))})",
            confidence=confidence,
            priority=max(s.metadata.priority for s in directional_signals),
            tags=['combined', 'correlation'],
            targets=list(set().union(*[s.metadata.targets for s in directional_signals]))
        )
        
        # Create combined signal
        tf_enum = next((tf for tf in Timeframe if tf.value == timeframe), Timeframe.H1)
        
        signal_data = {
            'component_signals': [s.id for s in directional_signals],
            'combination_method': 'avg',
            'direction_counts': {k.value: v for k, v in direction_counts.items()}
        }
        
        combined_signal = InterpretedSignal(
            symbol=symbol,
            timeframe=tf_enum,
            signal_type=SignalType.INTERPRETED,
            direction=max_direction,
            strength=strength,
            metadata=metadata,
            interpretation=f"Combined signal from {len(directional_signals)} correlated signals",
            indicators={
                'component_count': len(directional_signals),
                'avg_strength': avg_strength_value,
                'avg_confidence': avg_confidence_value,
                'signal_data': signal_data
            }
        )
        
        return combined_signal
    
    def _combine_signals_max(self, signals: List[Signal], 
                           symbol: str, 
                           timeframe: str,
                           context: Dict[str, Any]) -> Optional[Signal]:
        """Combine signals using max priority/confidence method"""
        # Filter signals by correlation threshold
        correlated_signals = self._filter_correlated_signals(signals)
        if len(correlated_signals) < 2:
            return None
        
        # Find signal with highest priority and confidence
        max_signal = max(
            correlated_signals,
            key=lambda s: (s.metadata.priority.value, s.metadata.confidence.value, s.strength.value)
        )
        
        # Create metadata
        sources = [s.metadata.source for s in correlated_signals]
        metadata = SignalMetadata(
            source=f"Combined({','.join(set(sources))})",
            confidence=max_signal.metadata.confidence,
            priority=max_signal.metadata.priority,
            tags=['combined', 'correlation', 'max_method'],
            targets=list(set().union(*[s.metadata.targets for s in correlated_signals]))
        )
        
        # Create combined signal (clone the max signal with new metadata)
        tf_enum = next((tf for tf in Timeframe if tf.value == timeframe), Timeframe.H1)
        
        signal_data = {
            'component_signals': [s.id for s in correlated_signals],
            'combination_method': 'max',
            'max_signal': max_signal.id
        }
        
        if isinstance(max_signal, InterpretedSignal):
            combined_signal = InterpretedSignal(
                symbol=symbol,
                timeframe=tf_enum,
                signal_type=SignalType.INTERPRETED,
                direction=max_signal.direction,
                strength=max_signal.strength,
                metadata=metadata,
                interpretation=f"Combined signal using max method from {len(correlated_signals)} correlated signals",
                indicators={
                    'component_count': len(correlated_signals),
                    'signal_data': signal_data,
                    **max_signal.indicators
                }
            )
        else:
            combined_signal = Signal(
                symbol=symbol,
                timeframe=tf_enum,
                signal_type=SignalType.INTERPRETED,
                direction=max_signal.direction,
                strength=max_signal.strength,
                metadata=metadata,
                data={
                    'component_count': len(correlated_signals),
                    'signal_data': signal_data,
                    'max_signal_data': max_signal.data if hasattr(max_signal, 'data') else {}
                }
            )
        
        return combined_signal
    
    def _combine_signals_weighted(self, signals: List[Signal], 
                               symbol: str, 
                               timeframe: str,
                               context: Dict[str, Any]) -> Optional[Signal]:
        """Combine signals using weighted method"""
        # Filter signals by correlation threshold
        correlated_signals = self._filter_correlated_signals(signals)
        if len(correlated_signals) < 2:
            return None
        
        # Calculate weighted direction
        direction_weights = {
            SignalDirection.BULLISH: 0,
            SignalDirection.BEARISH: 0,
            SignalDirection.NEUTRAL: 0
        }
        
        for signal in correlated_signals:
            # Calculate signal weight based on priority, confidence, and strength
            weight = signal.metadata.priority.value * signal.metadata.confidence.value * signal.strength.value
            direction_weights[signal.direction] += weight
        
        # Determine weighted direction
        max_direction = max(direction_weights.items(), key=lambda x: x[1])[0]
        
        # If too close, return neutral
        if (abs(direction_weights[SignalDirection.BULLISH] - direction_weights[SignalDirection.BEARISH]) / 
            (direction_weights[SignalDirection.BULLISH] + direction_weights[SignalDirection.BEARISH] + 0.001) < 0.1):
            max_direction = SignalDirection.NEUTRAL
        
        # Calculate weighted strength and confidence
        total_weight = sum(direction_weights.values())
        if total_weight == 0:
            return None
            
        # Calculate weighted values
        weighted_strength = 0
        weighted_confidence = 0
        
        for signal in correlated_signals:
            weight = signal.metadata.priority.value * signal.metadata.confidence.value * signal.strength.value
            weighted_strength += (signal.strength.value * weight)
            weighted_confidence += (signal.metadata.confidence.value * weight)
        
        weighted_strength /= total_weight
        weighted_confidence /= total_weight
        
        # Map to nearest enums
        strength_map = {
            1: SignalStrength.WEAK,
            2: SignalStrength.MODERATE,
            3: SignalStrength.STRONG,
            4: SignalStrength.VERY_STRONG
        }
        
        confidence_map = {
            1: SignalConfidence.SPECULATIVE,
            2: SignalConfidence.PROBABLE,
            3: SignalConfidence.LIKELY,
            4: SignalConfidence.CONFIRMED
        }
        
        # Round to nearest integer and limit to valid range
        strength_value = max(1, min(4, round(weighted_strength)))
        confidence_value = max(1, min(4, round(weighted_confidence)))
        
        strength = strength_map.get(strength_value, SignalStrength.MODERATE)
        confidence = confidence_map.get(confidence_value, SignalConfidence.PROBABLE)
        
        # Create metadata
        sources = [s.metadata.source for s in correlated_signals]
        metadata = SignalMetadata(
            source=f"Combined({','.join(set(sources))})",
            confidence=confidence,
            priority=max(s.metadata.priority for s in correlated_signals),
            tags=['combined', 'correlation', 'weighted_method'],
            targets=list(set().union(*[s.metadata.targets for s in correlated_signals]))
        )
        
        # Create combined signal
        tf_enum = next((tf for tf in Timeframe if tf.value == timeframe), Timeframe.H1)
        
        signal_data = {
            'component_signals': [s.id for s in correlated_signals],
            'combination_method': 'weighted',
            'direction_weights': {k.value: v for k, v in direction_weights.items()}
        }
        
        combined_signal = InterpretedSignal(
            symbol=symbol,
            timeframe=tf_enum,
            signal_type=SignalType.INTERPRETED,
            direction=max_direction,
            strength=strength,
            metadata=metadata,
            interpretation=f"Combined signal using weighted method from {len(correlated_signals)} correlated signals",
            indicators={
                'component_count': len(correlated_signals),
                'weighted_strength': weighted_strength,
                'weighted_confidence': weighted_confidence,
                'signal_data': signal_data
            }
        )
        
        return combined_signal
    
    def _filter_correlated_signals(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals by correlation threshold"""
        # Simple implementation - more advanced correlation detection could be implemented
        correlation_threshold = self.parameters.get('correlation_threshold', 0.7)
        
        # For now, assume all signals in the same timeframe are correlated enough
        # In a real implementation, you would analyze signal properties to measure correlation
        return signals