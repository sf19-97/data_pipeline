"""
Timeframe Signal Combiner

This component combines signals across timeframes for a single symbol.
"""

import copy
from typing import Dict, List, Any, Optional, Set, Tuple

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.signal.base import BaseSignalCombiner
from balance_breaker.src.signal.signal_models import (
    Signal, InterpretedSignal, SignalMetadata, Timeframe,
    SignalType, SignalDirection, SignalStrength, SignalConfidence
)

@implements("SignalCombiner")
class TimeframeCombiner(BaseSignalCombiner):
    """
    Combines signals across timeframes for a single symbol
    
    Parameters:
    -----------
    min_timeframes : int
        Minimum number of timeframes required to generate a combined signal (default: 2)
    weight_higher_timeframes : bool
        Whether to assign more weight to higher timeframes (default: True)
    timeframe_hierarchy : List[str]
        Hierarchy of timeframes from highest to lowest (default: standard hierarchy)
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'min_timeframes': 2,
            'weight_higher_timeframes': True,
            'timeframe_hierarchy': ['1M', '1w', '1d', '4h', '1h', '30m', '15m', '5m', '1m']
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
            min_timeframes = self.parameters.get('min_timeframes', 2)
            
            if len(signals) < min_timeframes:
                return []
            
            # Group signals by symbol
            symbol_groups = {}
            for signal in signals:
                if signal.symbol not in symbol_groups:
                    symbol_groups[signal.symbol] = {}
                    
                # Group by timeframe
                tf = signal.timeframe.value
                if tf not in symbol_groups[signal.symbol]:
                    symbol_groups[signal.symbol][tf] = []
                
                symbol_groups[signal.symbol][tf].append(signal)
            
            combined_signals = []
            
            # Process each symbol
            for symbol, timeframe_groups in symbol_groups.items():
                # Check if there are enough timeframes
                if len(timeframe_groups) < min_timeframes:
                    continue
                
                # Combine signals for this symbol
                combined_signal = self._combine_timeframe_signals(symbol, timeframe_groups, context)
                if combined_signal:
                    combined_signals.append(combined_signal)
            
            return combined_signals
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'signal_count': len(signals)},
                subsystem='signal',
                component='TimeframeCombiner'
            )
            return []
    
    def can_combine(self, signals: List[Signal]) -> bool:
        """Check if combiner can work with the given signals
        
        Args:
            signals: Signals to check
            
        Returns:
            True if signals can be combined, False otherwise
        """
        # Need at least min_timeframes timeframes for any symbol
        min_timeframes = self.parameters.get('min_timeframes', 2)
        
        # Group signals by symbol and count timeframes
        symbol_tf_counts = {}
        for signal in signals:
            if signal.symbol not in symbol_tf_counts:
                symbol_tf_counts[signal.symbol] = set()
            
            symbol_tf_counts[signal.symbol].add(signal.timeframe.value)
        
        return any(len(timeframes) >= min_timeframes for timeframes in symbol_tf_counts.values())
    
    def _combine_timeframe_signals(self, symbol: str, 
                                 timeframe_groups: Dict[str, List[Signal]],
                                 context: Dict[str, Any]) -> Optional[Signal]:
        """Combine signals across timeframes for a single symbol"""
        # Get parameters
        weight_higher_timeframes = self.parameters.get('weight_higher_timeframes', True)
        timeframe_hierarchy = self.parameters.get('timeframe_hierarchy', 
                                               ['1M', '1w', '1d', '4h', '1h', '30m', '15m', '5m', '1m'])
        
        # For each timeframe, select the strongest signal in the same direction
        selected_signals = []
        for tf in timeframe_hierarchy:
            if tf not in timeframe_groups:
                continue
                
            # Group by direction
            direction_groups = {
                SignalDirection.BULLISH: [],
                SignalDirection.BEARISH: [],
                SignalDirection.NEUTRAL: []
            }
            
            for signal in timeframe_groups[tf]:
                direction_groups[signal.direction].append(signal)
            
            # Add the strongest signal from each direction
            for direction, dir_signals in direction_groups.items():
                if not dir_signals:
                    continue
                    
                strongest = max(
                    dir_signals,
                    key=lambda s: (s.strength.value, s.metadata.confidence.value)
                )
                
                selected_signals.append(strongest)
        
        # Now combine selected signals
        if len(selected_signals) < 2:
            return None
        
        # Calculate direction weights
        direction_weights = {
            SignalDirection.BULLISH: 0,
            SignalDirection.BEARISH: 0,
            SignalDirection.NEUTRAL: 0
        }
        
        for signal in selected_signals:
            # Calculate weight based on timeframe and signal properties
            tf_weight = 1
            if weight_higher_timeframes:
                # Higher weight for higher timeframes
                tf_index = timeframe_hierarchy.index(signal.timeframe.value)
                tf_weight = len(timeframe_hierarchy) - tf_index
            
            # Final weight is timeframe weight times signal properties
            weight = tf_weight * signal.strength.value * signal.metadata.confidence.value
            direction_weights[signal.direction] += weight
        
        # Determine weighted direction
        max_direction = max(direction_weights.items(), key=lambda x: x[1])[0]
        
        # If neutral is strong enough, use it
        neutral_threshold = 0.3 * sum(direction_weights.values())
        if direction_weights[SignalDirection.NEUTRAL] >= neutral_threshold:
            max_direction = SignalDirection.NEUTRAL
        
        # Select final timeframe
        # Use the highest timeframe with a signal in the winning direction
        final_timeframe = None
        for tf in timeframe_hierarchy:
            if tf not in timeframe_groups:
                continue
                
            # Check if this timeframe has a signal in the winning direction
            matching_signals = [s for s in timeframe_groups[tf] if s.direction == max_direction]
            if matching_signals:
                final_timeframe = tf
                break
        
        if not final_timeframe:
            # Fallback to highest available timeframe
            for tf in timeframe_hierarchy:
                if tf in timeframe_groups:
                    final_timeframe = tf
                    break
        
        if not final_timeframe:
            return None
        
        # Calculate strength and confidence
        # Weight the signals in the winning direction
        directional_signals = [s for s in selected_signals if s.direction == max_direction]
        if not directional_signals:
            return None
            
        # Calculate weighted strength and confidence
        total_weight = sum(direction_weights.values())
        
        weighted_strength = sum(s.strength.value * (s.metadata.confidence.value / total_weight) 
                              for s in directional_signals)
        
        weighted_confidence = sum(s.metadata.confidence.value * (s.strength.value / total_weight)
                               for s in directional_signals)
        
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
        
        # Round to nearest integer
        strength_value = max(1, min(4, round(weighted_strength)))
        confidence_value = max(1, min(4, round(weighted_confidence)))
        
        strength = strength_map.get(strength_value, SignalStrength.MODERATE)
        confidence = confidence_map.get(confidence_value, SignalConfidence.PROBABLE)
        
        # Create metadata
        sources = [s.metadata.source for s in selected_signals]
        all_timeframes = [s.timeframe.value for s in selected_signals]
        
        metadata = SignalMetadata(
            source=f"MultiFTF({','.join(set(sources))})",
            confidence=confidence,
            priority=max(s.metadata.priority for s in selected_signals),
            tags=['combined', 'multi_timeframe', 'timeframe_' + final_timeframe],
            targets=list(set().union(*[s.metadata.targets for s in selected_signals]))
        )
        
        # Create combined signal
        tf_enum = next((tf for tf in Timeframe if tf.value == final_timeframe), Timeframe.H1)
        
        signal_data = {
            'component_signals': [s.id for s in selected_signals],
            'timeframes': all_timeframes,
            'direction_weights': {k.value: v for k, v in direction_weights.items()}
        }
        
        combined_signal = InterpretedSignal(
            symbol=symbol,
            timeframe=tf_enum,
            signal_type=SignalType.INTERPRETED,
            direction=max_direction,
            strength=strength,
            metadata=metadata,
            interpretation=f"Multi-timeframe signal from {len(all_timeframes)} timeframes",
            indicators={
                'component_count': len(selected_signals),
                'timeframe_count': len(set(all_timeframes)),
                'weighted_strength': weighted_strength,
                'weighted_confidence': weighted_confidence,
                'signal_data': signal_data
            }
        )
        
        return combined_signal