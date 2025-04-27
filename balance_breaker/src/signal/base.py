"""
Base Components for Signal Subsystem

This module defines the base classes for all signal components.
These base classes implement the interface contracts from the core framework.
"""

import logging
from abc import abstractmethod
from typing import Dict, Any, Optional, List, Set, Union
from datetime import datetime, timedelta

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.core.error_handling import ErrorHandler
from balance_breaker.src.core.parameter_manager import ParameterizedComponent
from balance_breaker.src.signal.signal_interfaces import (
    SignalComponent, SignalGenerator, SignalProcessor, 
    SignalFilter, SignalCombiner
)
from balance_breaker.src.signal.signal_models import (
    Signal, SignalMetadata, Timeframe, 
    SignalConfidence, SignalPriority
)


class BaseSignalComponent(ParameterizedComponent, SignalComponent):
    """Base class for all signal components
    
    Extends ParameterizedComponent from the core framework.
    Implements the SignalComponent interface.
    """
    
    def __init__(self, parameters=None):
        """Initialize with optional parameters
        
        Args:
            parameters: Dictionary of component parameters
        """
        super().__init__(parameters or {})
        
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Set up error handler
        self.error_handler = ErrorHandler()
    
    @property
    @abstractmethod
    def component_type(self) -> str:
        """Return component type identifier"""
        pass


@implements("SignalGenerator")
class BaseSignalGenerator(BaseSignalComponent):
    """Base class for signal generator components
    
    Signal generators create signals from input data.
    """
    
    @property
    def component_type(self) -> str:
        """Return component type identifier"""
        return 'generator'
    
    @abstractmethod
    def generate_signals(self, data: Any, context: Dict[str, Any]) -> List[Signal]:
        """Generate signals from data
        
        Args:
            data: Input data (typically from the data pipeline)
            context: Generation context with parameters
            
        Returns:
            List of generated signals
        """
        pass
    
    @abstractmethod
    def can_generate(self, data: Any, context: Dict[str, Any]) -> bool:
        """Check if generator can produce signals from the given data
        
        Args:
            data: Input data to check
            context: Generation context
            
        Returns:
            True if generator can produce signals, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def supported_timeframes(self) -> Set[Timeframe]:
        """Get supported timeframes for this generator
        
        Returns:
            Set of supported timeframes
        """
        pass
    
    @property
    @abstractmethod
    def required_data_types(self) -> Set[str]:
        """Get required data types for this generator
        
        Returns:
            Set of required data type strings
        """
        pass
    
    def validate_data(self, data: Any, context: Dict[str, Any]) -> bool:
        """Validate input data
        
        Args:
            data: Input data to validate
            context: Validation context
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check if data is None
        if data is None:
            self.logger.warning("Cannot generate signals from None data")
            return False
        
        # Get timeframe from context
        timeframe = context.get('timeframe')
        if timeframe and Timeframe(timeframe) not in self.supported_timeframes:
            self.logger.warning(f"Timeframe {timeframe} not supported by this generator")
            return False
        
        # Additional validation can be implemented in subclasses
        return True
    
    def create_signal_metadata(self, context: Dict[str, Any]) -> SignalMetadata:
        """Create metadata for a signal
        
        Args:
            context: Context with metadata information
            
        Returns:
            SignalMetadata object
        """
        # Extract metadata from context or use defaults
        return SignalMetadata(
            source=self.name,
            timestamp=datetime.now(),
            targets=context.get('targets', []),
            lifespan=context.get('lifespan'),
            confidence=context.get('confidence', SignalConfidence.PROBABLE),
            priority=context.get('priority', SignalPriority.MEDIUM),
            tags=context.get('tags', []),
            context={
                'generator_parameters': self.parameters,
                'generation_context': {k: v for k, v in context.items() 
                                      if k not in ['targets', 'lifespan', 'confidence', 'priority', 'tags']}
            }
        )


@implements("SignalProcessor")
class BaseSignalProcessor(BaseSignalComponent):
    """Base class for signal processor components
    
    Signal processors transform signals in various ways.
    """
    
    @property
    def component_type(self) -> str:
        """Return component type identifier"""
        return 'processor'
    
    @abstractmethod
    def process_signals(self, signals: List[Signal], context: Dict[str, Any]) -> List[Signal]:
        """Process signals
        
        Args:
            signals: Input signals to process
            context: Processing context
            
        Returns:
            Processed signals
        """
        pass
    
    @abstractmethod
    def can_process(self, signal: Signal) -> bool:
        """Check if processor can handle the given signal
        
        Args:
            signal: Signal to check
            
        Returns:
            True if processor can handle the signal, False otherwise
        """
        pass


@implements("SignalFilter")
class BaseSignalFilter(BaseSignalComponent):
    """Base class for signal filter components
    
    Signal filters select or reject signals based on criteria.
    """
    
    @property
    def component_type(self) -> str:
        """Return component type identifier"""
        return 'filter'
    
    def filter_signals(self, signals: List[Signal], context: Dict[str, Any]) -> List[Signal]:
        """Filter signals
        
        Args:
            signals: Input signals to filter
            context: Filtering context
            
        Returns:
            Filtered signals
        """
        return [signal for signal in signals if self.passes_filter(signal, context)]
    
    @abstractmethod
    def passes_filter(self, signal: Signal, context: Dict[str, Any]) -> bool:
        """Check if a single signal passes the filter
        
        Args:
            signal: Signal to check
            context: Filtering context
            
        Returns:
            True if signal passes filter, False otherwise
        """
        pass


@implements("SignalCombiner")
class BaseSignalCombiner(BaseSignalComponent):
    """Base class for signal combiner components
    
    Signal combiners merge multiple signals into new composite signals.
    """
    
    @property
    def component_type(self) -> str:
        """Return component type identifier"""
        return 'combiner'
    
    @abstractmethod
    def combine_signals(self, signals: List[Signal], context: Dict[str, Any]) -> List[Signal]:
        """Combine signals into new signals
        
        Args:
            signals: Input signals to combine
            context: Combination context
            
        Returns:
            Combined signals
        """
        pass
    
    @abstractmethod
    def can_combine(self, signals: List[Signal]) -> bool:
        """Check if combiner can work with the given signals
        
        Args:
            signals: Signals to check
            
        Returns:
            True if signals can be combined, False otherwise
        """
        pass