"""
Signal Interface Contracts

This module defines the core interfaces for the signal subsystem.
These interfaces provide contracts that components must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Set, Union
from datetime import datetime, timedelta

from balance_breaker.src.core.interface_registry import interface
from balance_breaker.src.signal.signal_models import Signal, SignalGroup, Timeframe


@interface
class SignalComponent(ABC):
    """
    Base interface for all signal components
    
    All components in the signal subsystem must implement this interface
    to ensure compatibility with the signal orchestrator.
    """
    
    @property
    @abstractmethod
    def component_type(self) -> str:
        """
        Return component type identifier
        
        Returns:
            String identifier for the component type
            (e.g., 'generator', 'processor', 'filter')
        """
        pass
    
    @property
    def name(self) -> str:
        """
        Return component name
        
        Returns:
            Name of the component (defaults to class name)
        """
        return self.__class__.__name__


@interface
class SignalGenerator(SignalComponent):
    """
    Interface for signal generation components
    
    Signal generators are responsible for creating signals from data.
    """
    
    @abstractmethod
    def generate_signals(self, data: Any, context: Dict[str, Any]) -> List[Signal]:
        """
        Generate signals from data
        
        Args:
            data: Input data (typically from the data pipeline)
            context: Generation context with parameters
            
        Returns:
            List of generated signals
        """
        pass
    
    @abstractmethod
    def can_generate(self, data: Any, context: Dict[str, Any]) -> bool:
        """
        Check if generator can produce signals from the given data
        
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
        """
        Get supported timeframes for this generator
        
        Returns:
            Set of supported timeframes
        """
        pass
    
    @property
    @abstractmethod
    def required_data_types(self) -> Set[str]:
        """
        Get required data types for this generator
        
        Returns:
            Set of required data type strings
        """
        pass


@interface
class SignalProcessor(SignalComponent):
    """
    Interface for signal processing components
    
    Signal processors transform signals in various ways.
    """
    
    @abstractmethod
    def process_signals(self, signals: List[Signal], context: Dict[str, Any]) -> List[Signal]:
        """
        Process signals
        
        Args:
            signals: Input signals to process
            context: Processing context
            
        Returns:
            Processed signals
        """
        pass
    
    @abstractmethod
    def can_process(self, signal: Signal) -> bool:
        """
        Check if processor can handle the given signal
        
        Args:
            signal: Signal to check
            
        Returns:
            True if processor can handle the signal, False otherwise
        """
        pass


@interface
class SignalFilter(SignalComponent):
    """
    Interface for signal filtering components
    
    Signal filters select or reject signals based on criteria.
    """
    
    @abstractmethod
    def filter_signals(self, signals: List[Signal], context: Dict[str, Any]) -> List[Signal]:
        """
        Filter signals
        
        Args:
            signals: Input signals to filter
            context: Filtering context
            
        Returns:
            Filtered signals
        """
        pass
    
    @abstractmethod
    def passes_filter(self, signal: Signal, context: Dict[str, Any]) -> bool:
        """
        Check if a single signal passes the filter
        
        Args:
            signal: Signal to check
            context: Filtering context
            
        Returns:
            True if signal passes filter, False otherwise
        """
        pass


@interface
class SignalCombiner(SignalComponent):
    """
    Interface for signal combination components
    
    Signal combiners merge multiple signals into new composite signals.
    """
    
    @abstractmethod
    def combine_signals(self, signals: List[Signal], context: Dict[str, Any]) -> List[Signal]:
        """
        Combine signals into new signals
        
        Args:
            signals: Input signals to combine
            context: Combination context
            
        Returns:
            Combined signals
        """
        pass
    
    @abstractmethod
    def can_combine(self, signals: List[Signal]) -> bool:
        """
        Check if combiner can work with the given signals
        
        Args:
            signals: Signals to check
            
        Returns:
            True if signals can be combined, False otherwise
        """
        pass


@interface
class SignalRegistry(ABC):
    """
    Interface for signal registry components
    
    Signal registries manage signal registration, lookup, and lifecycle.
    """
    
    @abstractmethod
    def register_signal(self, signal: Signal) -> bool:
        """
        Register a signal in the registry
        
        Args:
            signal: Signal to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_signal(self, signal_id: str) -> Optional[Signal]:
        """
        Get a signal by ID
        
        Args:
            signal_id: ID of the signal to retrieve
            
        Returns:
            Signal if found, None otherwise
        """
        pass
    
    @abstractmethod
    def remove_signal(self, signal_id: str) -> bool:
        """
        Remove a signal from the registry
        
        Args:
            signal_id: ID of the signal to remove
            
        Returns:
            True if removal was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def find_signals(self, criteria: Dict[str, Any]) -> List[Signal]:
        """
        Find signals matching criteria
        
        Args:
            criteria: Dictionary of search criteria
            
        Returns:
            List of matching signals
        """
        pass
    
    @abstractmethod
    def clean_expired_signals(self) -> int:
        """
        Remove expired signals from the registry
        
        Returns:
            Number of signals removed
        """
        pass


@interface
class SignalConsumer(ABC):
    """
    Interface for components that consume signals
    
    Signal consumers receive and act upon signals.
    """
    
    @abstractmethod
    def consume_signal(self, signal: Signal) -> bool:
        """
        Consume a signal
        
        Args:
            signal: Signal to consume
            
        Returns:
            True if signal was successfully consumed, False otherwise
        """
        pass
    
    @abstractmethod
    def can_consume(self, signal: Signal) -> bool:
        """
        Check if consumer can handle the given signal
        
        Args:
            signal: Signal to check
            
        Returns:
            True if consumer can handle the signal, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def accepted_signal_types(self) -> Set[str]:
        """
        Get accepted signal types
        
        Returns:
            Set of accepted signal type strings
        """
        pass