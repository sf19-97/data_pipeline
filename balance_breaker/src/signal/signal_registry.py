"""
Signal Registry

This module provides the registry for managing signals throughout their lifecycle.
It handles signal storage, retrieval, querying, and relationship tracking.
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime, timedelta
import copy

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.core.error_handling import ErrorHandler, BalanceBreakerError, ErrorSeverity, ErrorCategory
from balance_breaker.src.core.parameter_manager import ParameterizedComponent
from balance_breaker.src.core.integration_tools import event_bus
from balance_breaker.src.signal.signal_interfaces import SignalRegistry
from balance_breaker.src.signal.signal_models import Signal, SignalGroup, SignalType


class SignalRegistryError(BalanceBreakerError):
    """Error in signal registry operations"""
    
    def __init__(self, 
                message: str, 
                component: str = "",
                severity: ErrorSeverity = ErrorSeverity.ERROR,
                context: Optional[Dict[str, Any]] = None,
                original_exception: Optional[Exception] = None):
        super().__init__(
            message=message, 
            subsystem="signal",
            component=component,
            severity=severity,
            category=ErrorCategory.EXECUTION,
            context=context,
            original_exception=original_exception
        )


@implements("SignalRegistry")
class InMemorySignalRegistry(ParameterizedComponent, SignalRegistry):
    """
    In-memory implementation of the signal registry
    
    This registry stores signals in memory and provides methods for
    signal registration, retrieval, and lifecycle management.
    
    Parameters:
    -----------
    auto_clean_interval : int
        Interval in seconds for automatic cleaning of expired signals (default: 300)
    enable_auto_clean : bool
        Whether to enable automatic cleaning of expired signals (default: True)
    max_signal_age : int
        Maximum age in seconds for signals without explicit expiration (default: 86400)
    default_lifespan : int
        Default lifespan in seconds for signals without explicit lifespan (default: 3600)
    max_signals : int
        Maximum number of signals to store (default: 10000)
    """
    
    def __init__(self, parameters=None):
        """Initialize signal registry with optional parameters"""
        # Define default parameters
        default_params = {
            'auto_clean_interval': 300,  # 5 minutes
            'enable_auto_clean': True,
            'max_signal_age': 86400,     # 24 hours
            'default_lifespan': 3600,    # 1 hour
            'max_signals': 10000         # Maximum signals to store
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
        
        # Set up logging and error handling
        self.logger = logging.getLogger(__name__)
        self.error_handler = ErrorHandler(self.logger)
        
        # Storage for signals and relationships
        self._signals: Dict[str, Signal] = {}
        self._signal_groups: Dict[str, SignalGroup] = {}
        
        # Indices for efficient lookups
        self._symbol_index: Dict[str, Set[str]] = {}  # symbol -> set of signal IDs
        self._timeframe_index: Dict[str, Set[str]] = {}  # timeframe -> set of signal IDs
        self._type_index: Dict[str, Set[str]] = {}  # signal_type -> set of signal IDs
        self._source_index: Dict[str, Set[str]] = {}  # source -> set of signal IDs
        self._tag_index: Dict[str, Set[str]] = {}  # tag -> set of signal IDs
        
        # Thread lock for thread safety
        self._lock = threading.RLock()
        
        # Auto-clean thread
        self._auto_clean_thread = None
        self._stop_auto_clean = threading.Event()
        
        # Start auto-clean thread if enabled
        if self.parameters.get('enable_auto_clean', True):
            self._start_auto_clean()
    
    def _start_auto_clean(self) -> None:
        """Start the auto-clean thread"""
        if self._auto_clean_thread is not None and self._auto_clean_thread.is_alive():
            return  # Already running
        
        def auto_clean_worker():
            interval = self.parameters.get('auto_clean_interval', 300)
            while not self._stop_auto_clean.is_set():
                time.sleep(interval)
                try:
                    self.clean_expired_signals()
                except Exception as e:
                    self.error_handler.handle_error(
                        e,
                        context={'operation': 'auto_clean'},
                        subsystem='signal',
                        component='InMemorySignalRegistry'
                    )
        
        self._auto_clean_thread = threading.Thread(target=auto_clean_worker, daemon=True)
        self._auto_clean_thread.start()
        self.logger.info("Started auto-clean thread")
    
    def register_signal(self, signal: Signal) -> bool:
        """
        Register a signal in the registry
        
        Args:
            signal: Signal to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            with self._lock:
                # Check if registry is full
                if len(self._signals) >= self.parameters.get('max_signals', 10000):
                    # Remove oldest signals to make room
                    self._remove_oldest_signals(1)
                
                # Set default expiration if not set
                if signal.expiration_time is None:
                    default_lifespan = timedelta(seconds=self.parameters.get('default_lifespan', 3600))
                    signal.expiration_time = datetime.now() + default_lifespan
                
                # Make a copy to avoid external modification
                signal_copy = copy.deepcopy(signal)
                
                # Store signal
                self._signals[signal.id] = signal_copy
                
                # Update indices
                self._update_indices_for_signal(signal_copy)
                
                # Update relationship references if needed
                for parent_id in signal.parent_signals:
                    if parent_id in self._signals:
                        parent = self._signals[parent_id]
                        if signal.id not in parent.child_signals:
                            parent.child_signals.append(signal.id)
                
                # Publish signal registration event
                event_bus.publish('signal_registered', {
                    'signal_id': signal.id,
                    'signal_type': signal.signal_type.value,
                    'symbol': signal.symbol,
                    'timeframe': signal.timeframe.value
                })
                
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'signal_id': getattr(signal, 'id', None)},
                subsystem='signal',
                component='InMemorySignalRegistry'
            )
            return False
    
    def get_signal(self, signal_id: str) -> Optional[Signal]:
        """
        Get a signal by ID
        
        Args:
            signal_id: ID of the signal to retrieve
            
        Returns:
            Signal if found, None otherwise
        """
        with self._lock:
            # Return a copy to avoid external modification
            if signal_id in self._signals:
                return copy.deepcopy(self._signals[signal_id])
            return None
    
    def remove_signal(self, signal_id: str) -> bool:
        """
        Remove a signal from the registry
        
        Args:
            signal_id: ID of the signal to remove
            
        Returns:
            True if removal was successful, False otherwise
        """
        try:
            with self._lock:
                if signal_id not in self._signals:
                    return False
                
                # Get signal
                signal = self._signals[signal_id]
                
                # Remove from indices
                self._remove_from_indices(signal)
                
                # Update relationships
                for parent_id in signal.parent_signals:
                    if parent_id in self._signals:
                        parent = self._signals[parent_id]
                        if signal_id in parent.child_signals:
                            parent.child_signals.remove(signal_id)
                
                for child_id in signal.child_signals:
                    if child_id in self._signals:
                        child = self._signals[child_id]
                        if signal_id in child.parent_signals:
                            child.parent_signals.remove(signal_id)
                
                # Remove signal
                del self._signals[signal_id]
                
                # Publish signal removal event
                event_bus.publish('signal_removed', {
                    'signal_id': signal_id
                })
                
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'signal_id': signal_id},
                subsystem='signal',
                component='InMemorySignalRegistry'
            )
            return False
    
    def find_signals(self, criteria: Dict[str, Any]) -> List[Signal]:
        """
        Find signals matching criteria
        
        Args:
            criteria: Dictionary of search criteria
            
        Returns:
            List of matching signals
        """
        try:
            with self._lock:
                # Start with all signals
                matching_ids = set(self._signals.keys())
                
                # Apply criteria to narrow down results
                if 'symbol' in criteria:
                    symbol = criteria['symbol']
                    symbol_ids = self._symbol_index.get(symbol, set())
                    matching_ids &= symbol_ids
                
                if 'timeframe' in criteria:
                    timeframe = criteria['timeframe']
                    timeframe_value = timeframe.value if hasattr(timeframe, 'value') else timeframe
                    timeframe_ids = self._timeframe_index.get(timeframe_value, set())
                    matching_ids &= timeframe_ids
                
                if 'signal_type' in criteria:
                    signal_type = criteria['signal_type']
                    type_value = signal_type.value if hasattr(signal_type, 'value') else signal_type
                    type_ids = self._type_index.get(type_value, set())
                    matching_ids &= type_ids
                
                if 'source' in criteria:
                    source = criteria['source']
                    source_ids = self._source_index.get(source, set())
                    matching_ids &= source_ids
                
                if 'tags' in criteria:
                    tags = criteria['tags']
                    if isinstance(tags, str):
                        tags = [tags]
                    
                    tag_ids = set()
                    for tag in tags:
                        tag_ids |= self._tag_index.get(tag, set())
                    
                    matching_ids &= tag_ids
                
                if 'parent_id' in criteria:
                    parent_id = criteria['parent_id']
                    parent_ids = set()
                    for signal_id, signal in self._signals.items():
                        if parent_id in signal.parent_signals:
                            parent_ids.add(signal_id)
                    
                    matching_ids &= parent_ids
                
                if 'child_id' in criteria:
                    child_id = criteria['child_id']
                    child_ids = set()
                    for signal_id, signal in self._signals.items():
                        if child_id in signal.child_signals:
                            child_ids.add(signal_id)
                    
                    matching_ids &= child_ids
                
                if 'direction' in criteria:
                    direction = criteria['direction']
                    direction_value = direction.value if hasattr(direction, 'value') else direction
                    direction_ids = set()
                    for signal_id, signal in self._signals.items():
                        if signal.direction.value == direction_value:
                            direction_ids.add(signal_id)
                    
                    matching_ids &= direction_ids
                
                if 'min_strength' in criteria:
                    min_strength = criteria['min_strength']
                    strength_value = min_strength.value if hasattr(min_strength, 'value') else min_strength
                    strength_ids = set()
                    for signal_id, signal in self._signals.items():
                        if signal.strength.value >= strength_value:
                            strength_ids.add(signal_id)
                    
                    matching_ids &= strength_ids
                
                if 'min_priority' in criteria:
                    min_priority = criteria['min_priority']
                    priority_value = min_priority.value if hasattr(min_priority, 'value') else min_priority
                    priority_ids = set()
                    for signal_id, signal in self._signals.items():
                        if signal.metadata.priority.value >= priority_value:
                            priority_ids.add(signal_id)
                    
                    matching_ids &= priority_ids
                
                if 'min_confidence' in criteria:
                    min_confidence = criteria['min_confidence']
                    confidence_value = min_confidence.value if hasattr(min_confidence, 'value') else min_confidence
                    confidence_ids = set()
                    for signal_id, signal in self._signals.items():
                        if signal.metadata.confidence.value >= confidence_value:
                            confidence_ids.add(signal_id)
                    
                    matching_ids &= confidence_ids
                
                if 'active_only' in criteria and criteria['active_only']:
                    # Filter out expired signals
                    now = datetime.now()
                    active_ids = set()
                    for signal_id in matching_ids:
                        signal = self._signals[signal_id]
                        if signal.expiration_time is None or signal.expiration_time > now:
                            active_ids.add(signal_id)
                    
                    matching_ids = active_ids
                
                # Convert IDs to signals (return copies to avoid modification)
                return [copy.deepcopy(self._signals[signal_id]) for signal_id in matching_ids]
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'criteria': criteria},
                subsystem='signal',
                component='InMemorySignalRegistry'
            )
            return []
    
    def clean_expired_signals(self) -> int:
        """
        Remove expired signals from the registry
        
        Returns:
            Number of signals removed
        """
        try:
            with self._lock:
                now = datetime.now()
                max_age = timedelta(seconds=self.parameters.get('max_signal_age', 86400))
                
                to_remove = []
                for signal_id, signal in self._signals.items():
                    # Check for explicit expiration
                    if signal.expiration_time and signal.expiration_time <= now:
                        to_remove.append(signal_id)
                    # Check for max age
                    elif (now - signal.creation_time) > max_age:
                        to_remove.append(signal_id)
                
                # Remove signals
                for signal_id in to_remove:
                    self.remove_signal(signal_id)
                
                return len(to_remove)
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'operation': 'clean_expired_signals'},
                subsystem='signal',
                component='InMemorySignalRegistry'
            )
            return 0
    
    def register_signal_group(self, group: SignalGroup) -> bool:
        """
        Register a signal group in the registry
        
        Args:
            group: Signal group to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            with self._lock:
                # Make a copy to avoid external modification
                group_copy = copy.deepcopy(group)
                
                # Store group
                self._signal_groups[group.id] = group_copy
                
                # Register all signals in the group if they're not already registered
                for signal in group.signals:
                    if signal.id not in self._signals:
                        self.register_signal(signal)
                
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'group_id': getattr(group, 'id', None)},
                subsystem='signal',
                component='InMemorySignalRegistry'
            )
            return False
    
    def get_signal_group(self, group_id: str) -> Optional[SignalGroup]:
        """
        Get a signal group by ID
        
        Args:
            group_id: ID of the group to retrieve
            
        Returns:
            Signal group if found, None otherwise
        """
        with self._lock:
            if group_id in self._signal_groups:
                return copy.deepcopy(self._signal_groups[group_id])
            return None
    
    def get_related_signals(self, signal_id: str, include_parents: bool = True, 
                           include_children: bool = True) -> List[Signal]:
        """
        Get signals related to the given signal
        
        Args:
            signal_id: ID of the signal to find relations for
            include_parents: Whether to include parent signals
            include_children: Whether to include child signals
            
        Returns:
            List of related signals
        """
        try:
            with self._lock:
                if signal_id not in self._signals:
                    return []
                
                signal = self._signals[signal_id]
                related_ids = set()
                
                if include_parents:
                    related_ids.update(signal.parent_signals)
                
                if include_children:
                    related_ids.update(signal.child_signals)
                
                # Convert IDs to signals (return copies to avoid modification)
                result = []
                for related_id in related_ids:
                    if related_id in self._signals:
                        result.append(copy.deepcopy(self._signals[related_id]))
                
                return result
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'signal_id': signal_id},
                subsystem='signal',
                component='InMemorySignalRegistry'
            )
            return []
    
    def link_signals(self, parent_id: str, child_id: str) -> bool:
        """
        Create a parent-child relationship between signals
        
        Args:
            parent_id: ID of the parent signal
            child_id: ID of the child signal
            
        Returns:
            True if relationship was created, False otherwise
        """
        try:
            with self._lock:
                if parent_id not in self._signals or child_id not in self._signals:
                    return False
                
                parent = self._signals[parent_id]
                child = self._signals[child_id]
                
                # Update relationships
                if child_id not in parent.child_signals:
                    parent.child_signals.append(child_id)
                
                if parent_id not in child.parent_signals:
                    child.parent_signals.append(parent_id)
                
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'parent_id': parent_id, 'child_id': child_id},
                subsystem='signal',
                component='InMemorySignalRegistry'
            )
            return False
    
    def unlink_signals(self, parent_id: str, child_id: str) -> bool:
        """
        Remove a parent-child relationship between signals
        
        Args:
            parent_id: ID of the parent signal
            child_id: ID of the child signal
            
        Returns:
            True if relationship was removed, False otherwise
        """
        try:
            with self._lock:
                if parent_id not in self._signals or child_id not in self._signals:
                    return False
                
                parent = self._signals[parent_id]
                child = self._signals[child_id]
                
                # Update relationships
                if child_id in parent.child_signals:
                    parent.child_signals.remove(child_id)
                
                if parent_id in child.parent_signals:
                    child.parent_signals.remove(parent_id)
                
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'parent_id': parent_id, 'child_id': child_id},
                subsystem='signal',
                component='InMemorySignalRegistry'
            )
            return False
    
    def count_signals(self, filter_criteria: Optional[Dict[str, Any]] = None) -> int:
        """
        Count signals in the registry, optionally filtered by criteria
        
        Args:
            filter_criteria: Optional filtering criteria
            
        Returns:
            Count of matching signals
        """
        if filter_criteria:
            return len(self.find_signals(filter_criteria))
        
        with self._lock:
            return len(self._signals)
    
    def _update_indices_for_signal(self, signal: Signal) -> None:
        """
        Update all indices for a signal
        
        Args:
            signal: Signal to update indices for
        """
        # Symbol index
        if signal.symbol not in self._symbol_index:
            self._symbol_index[signal.symbol] = set()
        self._symbol_index[signal.symbol].add(signal.id)
        
        # Timeframe index
        timeframe_value = signal.timeframe.value
        if timeframe_value not in self._timeframe_index:
            self._timeframe_index[timeframe_value] = set()
        self._timeframe_index[timeframe_value].add(signal.id)
        
        # Type index
        type_value = signal.signal_type.value
        if type_value not in self._type_index:
            self._type_index[type_value] = set()
        self._type_index[type_value].add(signal.id)
        
        # Source index
        source = signal.metadata.source
        if source not in self._source_index:
            self._source_index[source] = set()
        self._source_index[source].add(signal.id)
        
        # Tag index
        for tag in signal.metadata.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(signal.id)
    
    def _remove_from_indices(self, signal: Signal) -> None:
        """
        Remove a signal from all indices
        
        Args:
            signal: Signal to remove from indices
        """
        # Symbol index
        if signal.symbol in self._symbol_index:
            self._symbol_index[signal.symbol].discard(signal.id)
        
        # Timeframe index
        timeframe_value = signal.timeframe.value
        if timeframe_value in self._timeframe_index:
            self._timeframe_index[timeframe_value].discard(signal.id)
        
        # Type index
        type_value = signal.signal_type.value
        if type_value in self._type_index:
            self._type_index[type_value].discard(signal.id)
        
        # Source index
        source = signal.metadata.source
        if source in self._source_index:
            self._source_index[source].discard(signal.id)
        
        # Tag index
        for tag in signal.metadata.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(signal.id)
    
    def _remove_oldest_signals(self, count: int) -> None:
        """
        Remove the oldest signals from the registry
        
        Args:
            count: Number of signals to remove
        """
        if not self._signals:
            return
        
        # Sort signals by creation time
        sorted_signals = sorted(
            self._signals.items(),
            key=lambda item: item[1].creation_time
        )
        
        # Remove oldest signals
        for i in range(min(count, len(sorted_signals))):
            signal_id, _ = sorted_signals[i]
            self.remove_signal(signal_id)
    
    def __del__(self):
        """Cleanup when registry is destroyed"""
        if self._auto_clean_thread and self._auto_clean_thread.is_alive():
            self._stop_auto_clean.set()
            self._auto_clean_thread.join(timeout=1.0)