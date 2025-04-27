"""
Signal Enricher Processor

This component enriches signals with additional market context and information.
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Set

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.signal.base import BaseSignalProcessor
from balance_breaker.src.signal.signal_models import (
    Signal, RawSignal, InterpretedSignal, ActionSignal,
    SignalMetadata, Timeframe, SignalType, 
    SignalDirection, SignalStrength, SignalConfidence
)

@implements("SignalProcessor")
class SignalEnricherProcessor(BaseSignalProcessor):
    """
    Enriches signals with additional market context and information
    
    Parameters:
    -----------
    market_data_key : str
        Key for market data in context (default: 'market_data')
    enrich_interpreted_signals : bool
        Whether to enrich interpreted signals (default: True)
    enrich_raw_signals : bool
        Whether to enrich raw signals (default: False)
    add_market_conditions : bool
        Whether to add market conditions (default: True)
    enrich_tags : bool
        Whether to add enrichment tags (default: True)
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'market_data_key': 'market_data',
            'enrich_interpreted_signals': True,
            'enrich_raw_signals': False,
            'add_market_conditions': True,
            'enrich_tags': True
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def process_signals(self, signals: List[Signal], context: Dict[str, Any]) -> List[Signal]:
        """Process signals by enriching with additional information
        
        Args:
            signals: Input signals to process
            context: Processing context with market data
            
        Returns:
            Enriched signals
        """
        try:
            if not signals:
                return []
                
            # Get market data from context
            market_data_key = self.parameters.get('market_data_key', 'market_data')
            market_data = context.get(market_data_key, {})
            
            # Process each signal
            enriched_signals = []
            for signal in signals:
                # Check if signal should be enriched based on type
                if (signal.signal_type == SignalType.INTERPRETED and 
                    not self.parameters.get('enrich_interpreted_signals', True)):
                    enriched_signals.append(signal)
                    continue
                    
                if (signal.signal_type == SignalType.RAW and 
                    not self.parameters.get('enrich_raw_signals', False)):
                    enriched_signals.append(signal)
                    continue
                
                # Enrich the signal
                enriched_signal = self._enrich_signal(signal, market_data, context)
                enriched_signals.append(enriched_signal)
            
            return enriched_signals
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'signal_count': len(signals)},
                subsystem='signal',
                component='SignalEnricherProcessor'
            )
            return signals  # Return original signals on error
    
    def can_process(self, signal: Signal) -> bool:
        """Check if processor can handle the given signal
        
        Args:
            signal: Signal to check
            
        Returns:
            True if processor can handle the signal, False otherwise
        """
        # Can process any signal type, but will only actually enrich
        # if the specific type is enabled in parameters
        return True
    
    def _enrich_signal(self, signal: Signal, market_data: Dict[str, Any], 
                      context: Dict[str, Any]) -> Signal:
        """Enrich a single signal with market context
        
        Args:
            signal: Signal to enrich
            market_data: Market data dictionary
            context: Processing context
            
        Returns:
            Enriched signal
        """
        # Create a copy to avoid modifying the original
        enriched = self._clone_signal(signal)
        
        # Get market data for the signal's symbol
        symbol_data = market_data.get(signal.symbol, {})
        
        # Add symbol metadata
        self._add_symbol_metadata(enriched, symbol_data, context)
        
        # Add market conditions if enabled
        if self.parameters.get('add_market_conditions', True):
            self._add_market_conditions(enriched, symbol_data, context)
        
        # Add enrichment tags if enabled
        if self.parameters.get('enrich_tags', True):
            self._add_enrichment_tags(enriched, symbol_data, context)
        
        # Add additional interpretation if it's an interpreted signal
        if isinstance(enriched, InterpretedSignal):
            self._enhance_interpretation(enriched, symbol_data, context)
        
        # Add additional action parameters if it's an action signal
        if isinstance(enriched, ActionSignal):
            self._enhance_action_parameters(enriched, symbol_data, context)
        
        return enriched
    
    def _clone_signal(self, signal: Signal) -> Signal:
        """Create a clone of a signal with the same properties
        
        Args:
            signal: Signal to clone
            
        Returns:
            Cloned signal
        """
        # Create a new signal of the same type with the same properties
        if isinstance(signal, ActionSignal):
            return ActionSignal(
                id=signal.id,
                symbol=signal.symbol,
                timeframe=signal.timeframe,
                signal_type=signal.signal_type,
                direction=signal.direction,
                strength=signal.strength,
                metadata=signal.metadata,
                creation_time=signal.creation_time,
                expiration_time=signal.expiration_time,
                parent_signals=signal.parent_signals,
                child_signals=signal.child_signals,
                data=signal.data.copy() if hasattr(signal, 'data') else {},
                action_type=signal.action_type,
                action_parameters=signal.action_parameters.copy(),
                urgency=signal.urgency
            )
        elif isinstance(signal, InterpretedSignal):
            return InterpretedSignal(
                id=signal.id,
                symbol=signal.symbol,
                timeframe=signal.timeframe,
                signal_type=signal.signal_type,
                direction=signal.direction,
                strength=signal.strength,
                metadata=signal.metadata,
                creation_time=signal.creation_time,
                expiration_time=signal.expiration_time,
                parent_signals=signal.parent_signals,
                child_signals=signal.child_signals,
                data=signal.data.copy() if hasattr(signal, 'data') else {},
                interpretation=signal.interpretation,
                indicators=signal.indicators.copy() if hasattr(signal, 'indicators') else {}
            )
        elif isinstance(signal, RawSignal):
            return RawSignal(
                id=signal.id,
                symbol=signal.symbol,
                timeframe=signal.timeframe,
                signal_type=signal.signal_type,
                direction=signal.direction,
                strength=signal.strength,
                metadata=signal.metadata,
                creation_time=signal.creation_time,
                expiration_time=signal.expiration_time,
                parent_signals=signal.parent_signals,
                child_signals=signal.child_signals,
                data=signal.data.copy() if hasattr(signal, 'data') else {}
            )
        else:
            return Signal(
                id=signal.id,
                symbol=signal.symbol,
                timeframe=signal.timeframe,
                signal_type=signal.signal_type,
                direction=signal.direction,
                strength=signal.strength,
                metadata=signal.metadata,
                creation_time=signal.creation_time,
                expiration_time=signal.expiration_time,
                parent_signals=signal.parent_signals,
                child_signals=signal.child_signals,
                data=signal.data.copy() if hasattr(signal, 'data') else {}
            )
    
    def _add_symbol_metadata(self, signal: Signal, symbol_data: Dict[str, Any], 
                           context: Dict[str, Any]) -> None:
        """Add symbol metadata to the signal
        
        Args:
            signal: Signal to enrich
            symbol_data: Market data for the symbol
            context: Processing context
        """
        # Get general symbol metadata from context
        symbol_metadata = context.get('symbol_metadata', {}).get(signal.symbol, {})
        
        if not hasattr(signal, 'data'):
            signal.data = {}
        
        # Add symbol metadata to signal data
        if symbol_metadata:
            if 'symbol_metadata' not in signal.data:
                signal.data['symbol_metadata'] = {}
                
            # Add basic symbol information
            signal.data['symbol_metadata'].update({
                'base_currency': symbol_metadata.get('base_currency', signal.symbol[:3]),
                'quote_currency': symbol_metadata.get('quote_currency', signal.symbol[3:]),
                'pip_value': symbol_metadata.get('pip_value', 0.0001),
                'is_major': symbol_metadata.get('is_major', False),
                'trading_hours': symbol_metadata.get('trading_hours', '24h')
            })
    
    def _add_market_conditions(self, signal: Signal, symbol_data: Dict[str, Any], 
                            context: Dict[str, Any]) -> None:
        """Add market condition information to the signal
        
        Args:
            signal: Signal to enrich
            symbol_data: Market data for the symbol
            context: Processing context
        """
        # Get current price data if available
        price_data = symbol_data.get('price', {})
        
        if not price_data:
            return
            
        if not hasattr(signal, 'data'):
            signal.data = {}
            
        # Create market conditions data object
        if 'market_conditions' not in signal.data:
            signal.data['market_conditions'] = {}
        
        # Add current price information
        signal.data['market_conditions'].update({
            'current_price': price_data.get('close', 0),
            'daily_range': price_data.get('high', 0) - price_data.get('low', 0),
            'daily_change_pct': price_data.get('daily_change_pct', 0)
        })
        
        # Add volatility information if available
        if 'volatility' in symbol_data:
            signal.data['market_conditions']['volatility'] = symbol_data['volatility']
        
        # Add trend information if available
        if 'trend' in symbol_data:
            signal.data['market_conditions']['trend'] = symbol_data['trend']
        
        # Add support/resistance levels if available
        if 'levels' in symbol_data:
            signal.data['market_conditions']['key_levels'] = symbol_data['levels']
    
    def _add_enrichment_tags(self, signal: Signal, symbol_data: Dict[str, Any], 
                          context: Dict[str, Any]) -> None:
        """Add enrichment tags to the signal
        
        Args:
            signal: Signal to enrich
            symbol_data: Market data for the symbol
            context: Processing context
        """
        # Add enriched tag
        if 'enriched' not in signal.metadata.tags:
            signal.metadata.tags.append('enriched')
        
        # Add market condition tags
        if 'trend' in symbol_data:
            trend = symbol_data['trend']
            if trend == 'bullish':
                signal.metadata.tags.append('bullish_trend')
            elif trend == 'bearish':
                signal.metadata.tags.append('bearish_trend')
            elif trend == 'ranging':
                signal.metadata.tags.append('ranging_market')
        
        # Add volatility tags
        if 'volatility' in symbol_data:
            volatility = symbol_data['volatility']
            if volatility == 'high':
                signal.metadata.tags.append('high_volatility')
            elif volatility == 'low':
                signal.metadata.tags.append('low_volatility')
        
        # Add market session tags
        if 'current_session' in context:
            session = context['current_session']
            signal.metadata.tags.append(f'session_{session.lower()}')
    
    def _enhance_interpretation(self, signal: InterpretedSignal, symbol_data: Dict[str, Any], 
                             context: Dict[str, Any]) -> None:
        """Enhance interpretation for interpreted signals
        
        Args:
            signal: Interpreted signal to enhance
            symbol_data: Market data for the symbol
            context: Processing context
        """
        # Add market context to interpretation if available
        market_context = ""
        
        if 'trend' in symbol_data:
            market_context += f" in a {symbol_data['trend']} trend"
        
        if 'volatility' in symbol_data:
            market_context += f" with {symbol_data['volatility']} volatility"
        
        if market_context and not signal.interpretation.endswith('.'):
            signal.interpretation += '.'
            
        if market_context:
            signal.interpretation += f" Market is currently{market_context}."
        
        # Add additional indicators if available
        if 'indicators' in symbol_data and signal.indicators:
            for indicator, value in symbol_data['indicators'].items():
                if indicator not in signal.indicators:
                    signal.indicators[indicator] = value
    
    def _enhance_action_parameters(self, signal: ActionSignal, symbol_data: Dict[str, Any], 
                           context: Dict[str, Any]) -> None:
        """Enhance action parameters for action signals
        
        Args:
            signal: Action signal to enhance
            symbol_data: Market data for the symbol
            context: Processing context
        """
        # Add current price if not already set
        if 'price' not in signal.action_parameters and 'close' in symbol_data.get('price', {}):
            signal.action_parameters['price'] = symbol_data['price']['close']
        
        # Add market context information that might be useful for risk management
        if 'atr' in symbol_data:
            signal.action_parameters['current_atr'] = symbol_data['atr']
            
        # Add volatility information if available
        if 'volatility' in symbol_data:
            signal.action_parameters['volatility'] = symbol_data['volatility']
            
        # Add key levels information if available
        if 'levels' in symbol_data:
            signal.action_parameters['key_levels'] = symbol_data['levels']