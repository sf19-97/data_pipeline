"""
Data Pipeline Registry

This module provides the registry for managing data throughout its lifecycle.
It handles data storage, retrieval, querying, and relationship tracking.
"""

import logging
import threading
import time
import uuid
import copy
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime, timedelta

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.core.error_handling import ErrorHandler, BalanceBreakerError, ErrorSeverity, ErrorCategory
from balance_breaker.src.core.parameter_manager import ParameterizedComponent
from balance_breaker.src.core.integration_tools import event_bus
from balance_breaker.src.data_pipeline.data_pipeline_interfaces import DataRegistry


class DataRegistryError(BalanceBreakerError):
    """Error in data registry operations"""
    
    def __init__(self, 
                message: str, 
                component: str = "",
                severity: ErrorSeverity = ErrorSeverity.ERROR,
                context: Optional[Dict[str, Any]] = None,
                original_exception: Optional[Exception] = None):
        super().__init__(
            message=message, 
            subsystem="data_pipeline",
            component=component,
            severity=severity,
            category=ErrorCategory.EXECUTION,
            context=context,
            original_exception=original_exception
        )


@implements("DataRegistry")
class InMemoryDataRegistry(ParameterizedComponent, DataRegistry):
    """
    In-memory implementation of the data registry
    
    This registry stores data in memory and provides methods for
    data registration, retrieval, and lifecycle management.
    
    Parameters:
    -----------
    auto_clean_interval : int
        Interval in seconds for automatic cleaning of expired data (default: 300)
    enable_auto_clean : bool
        Whether to enable automatic cleaning of expired data (default: True)
    max_data_age : int
        Maximum age in seconds for data without explicit expiration (default: 86400)
    default_lifespan : int
        Default lifespan in seconds for data without explicit lifespan (default: 3600)
    max_entries : int
        Maximum number of data entries to store (default: 10000)
    """
    
    def __init__(self, parameters=None):
        """Initialize data registry with optional parameters"""
        # Define default parameters
        default_params = {
            'auto_clean_interval': 300,  # 5 minutes
            'enable_auto_clean': True,
            'max_data_age': 86400,     # 24 hours
            'default_lifespan': 3600,    # 1 hour
            'max_entries': 10000         # Maximum data entries to store
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
        
        # Set up logging and error handling
        self.logger = logging.getLogger(__name__)
        self.error_handler = ErrorHandler(self.logger)
        
        # Storage for data and metadata
        self._data: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        # Storage for relationships
        self._relationships: Dict[str, Dict[str, List[str]]] = {}
        
        # Indices for efficient lookups
        self._type_index: Dict[str, Set[str]] = {}  # data_type -> set of data IDs
        self._source_index: Dict[str, Set[str]] = {}  # source -> set of data IDs
        self._tag_index: Dict[str, Set[str]] = {}  # tag -> set of data IDs
        self._pair_index: Dict[str, Set[str]] = {}  # pair -> set of data IDs
        self._time_index: Dict[str, Set[str]] = {}  # timeframe -> set of data IDs
        
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
                    self.clean_expired_data()
                except Exception as e:
                    self.error_handler.handle_error(
                        e,
                        context={'operation': 'auto_clean'},
                        subsystem='data_pipeline',
                        component='InMemoryDataRegistry'
                    )
        
        self._auto_clean_thread = threading.Thread(target=auto_clean_worker, daemon=True)
        self._auto_clean_thread.start()
        self.logger.info("Started auto-clean thread")
    
    def register_data(self, data: Any, metadata: Dict[str, Any]) -> str:
        """
        Register data in the registry
        
        Args:
            data: Data to register
            metadata: Metadata for the data
            
        Returns:
            Data ID
        """
        try:
            with self._lock:
                # Check if registry is full
                if len(self._data) >= self.parameters.get('max_entries', 10000):
                    # Remove oldest entries to make room
                    self._remove_oldest_entries(1)
                
                # Generate ID if not provided
                data_id = metadata.get('id', str(uuid.uuid4()))
                
                # Set creation time if not provided
                if 'creation_time' not in metadata:
                    metadata['creation_time'] = datetime.now()
                
                # Set expiration time if not provided
                if 'expiration_time' not in metadata:
                    default_lifespan = timedelta(seconds=self.parameters.get('default_lifespan', 3600))
                    metadata['expiration_time'] = metadata['creation_time'] + default_lifespan
                
                # Store data and metadata
                self._data[data_id] = copy.deepcopy(data)
                self._metadata[data_id] = copy.deepcopy(metadata)
                
                # Initialize relationships if needed
                if data_id not in self._relationships:
                    self._relationships[data_id] = {}
                
                # Update indices
                self._update_indices_for_data(data_id, metadata)
                
                # Publish data registration event
                event_bus.publish('data_registered', {
                    'data_id': data_id,
                    'data_type': metadata.get('data_type', 'unknown'),
                    'source': metadata.get('source', 'unknown')
                })
                
                return data_id
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'data_id': metadata.get('id', None)},
                subsystem='data_pipeline',
                component='InMemoryDataRegistry'
            )
            raise DataRegistryError(
                message=f"Failed to register data: {str(e)}",
                component='InMemoryDataRegistry',
                context={'metadata': metadata},
                original_exception=e
            )
    
    def get_data(self, data_id: str) -> Optional[Any]:
        """
        Get data by ID
        
        Args:
            data_id: ID of the data to retrieve
            
        Returns:
            Data if found, None otherwise
        """
        with self._lock:
            # Return a copy to avoid external modification
            if data_id in self._data:
                return copy.deepcopy(self._data[data_id])
            return None
    
    def find_data(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find data matching criteria
        
        Args:
            criteria: Dictionary of search criteria
            
        Returns:
            List of data entries with metadata
        """
        try:
            with self._lock:
                # Start with all data IDs
                matching_ids = set(self._data.keys())
                
                # Apply criteria to narrow down results
                if 'data_type' in criteria:
                    data_type = criteria['data_type']
                    type_ids = self._type_index.get(data_type, set())
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
                
                if 'pair' in criteria:
                    pair = criteria['pair']
                    pair_ids = self._pair_index.get(pair, set())
                    matching_ids &= pair_ids
                
                if 'timeframe' in criteria:
                    timeframe = criteria['timeframe']
                    time_ids = self._time_index.get(timeframe, set())
                    matching_ids &= time_ids
                
                if 'after_date' in criteria:
                    after_date = criteria['after_date']
                    after_ids = set()
                    for data_id, metadata in self._metadata.items():
                        creation_time = metadata.get('creation_time')
                        if creation_time and creation_time >= after_date:
                            after_ids.add(data_id)
                    
                    matching_ids &= after_ids
                
                if 'before_date' in criteria:
                    before_date = criteria['before_date']
                    before_ids = set()
                    for data_id, metadata in self._metadata.items():
                        creation_time = metadata.get('creation_time')
                        if creation_time and creation_time <= before_date:
                            before_ids.add(data_id)
                    
                    matching_ids &= before_ids
                
                if 'related_to' in criteria:
                    related_to = criteria['related_to']
                    relationship_type = criteria.get('relationship_type')
                    related_ids = set(self.get_related_data(related_to, relationship_type))
                    matching_ids &= related_ids
                
                if 'active_only' in criteria and criteria['active_only']:
                    # Filter out expired data
                    now = datetime.now()
                    active_ids = set()
                    for data_id in matching_ids:
                        metadata = self._metadata[data_id]
                        expiration_time = metadata.get('expiration_time')
                        if expiration_time is None or expiration_time > now:
                            active_ids.add(data_id)
                    
                    matching_ids = active_ids
                
                # Build result list
                results = []
                for data_id in matching_ids:
                    results.append({
                        'id': data_id,
                        'data': copy.deepcopy(self._data[data_id]),
                        'metadata': copy.deepcopy(self._metadata[data_id])
                    })
                
                return results
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'criteria': criteria},
                subsystem='data_pipeline',
                component='InMemoryDataRegistry'
            )
            return []
    
    def update_data(self, data_id: str, data: Any) -> bool:
        """
        Update existing data
        
        Args:
            data_id: ID of the data to update
            data: New data
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            with self._lock:
                if data_id not in self._data:
                    return False
                
                # Update data
                self._data[data_id] = copy.deepcopy(data)
                
                # Update metadata
                self._metadata[data_id]['last_updated'] = datetime.now()
                
                # Publish data updated event
                event_bus.publish('data_updated', {
                    'data_id': data_id
                })
                
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'data_id': data_id},
                subsystem='data_pipeline',
                component='InMemoryDataRegistry'
            )
            return False
    
    def remove_data(self, data_id: str) -> bool:
        """
        Remove data from the registry
        
        Args:
            data_id: ID of the data to remove
            
        Returns:
            True if removal was successful, False otherwise
        """
        try:
            with self._lock:
                if data_id not in self._data:
                    return False
                
                # Get metadata
                metadata = self._metadata[data_id]
                
                # Remove from indices
                self._remove_from_indices(data_id, metadata)
                
                # Remove relationship references
                self._remove_relationships(data_id)
                
                # Remove data and metadata
                del self._data[data_id]
                del self._metadata[data_id]
                
                # Publish data removed event
                event_bus.publish('data_removed', {
                    'data_id': data_id
                })
                
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'data_id': data_id},
                subsystem='data_pipeline',
                component='InMemoryDataRegistry'
            )
            return False
    
    def create_relationship(self, source_id: str, target_id: str, relationship_type: str) -> bool:
        """
        Create relationship between data products
        
        Args:
            source_id: Source data ID
            target_id: Target data ID
            relationship_type: Type of relationship
            
        Returns:
            True if relationship was created, False otherwise
        """
        try:
            with self._lock:
                if source_id not in self._data or target_id not in self._data:
                    return False
                
                # Initialize relationships if needed
                if source_id not in self._relationships:
                    self._relationships[source_id] = {}
                
                if target_id not in self._relationships:
                    self._relationships[target_id] = {}
                
                # Add relationship
                if relationship_type not in self._relationships[source_id]:
                    self._relationships[source_id][relationship_type] = []
                
                if target_id not in self._relationships[source_id][relationship_type]:
                    self._relationships[source_id][relationship_type].append(target_id)
                
                # Add inverse relationship
                inverse_type = f"inverse_{relationship_type}"
                if inverse_type not in self._relationships[target_id]:
                    self._relationships[target_id][inverse_type] = []
                
                if source_id not in self._relationships[target_id][inverse_type]:
                    self._relationships[target_id][inverse_type].append(source_id)
                
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'source_id': source_id, 'target_id': target_id},
                subsystem='data_pipeline',
                component='InMemoryDataRegistry'
            )
            return False
    
    def get_related_data(self, data_id: str, relationship_type: Optional[str] = None) -> List[str]:
        """
        Get related data IDs
        
        Args:
            data_id: Data ID to find relations for
            relationship_type: Optional filter by relationship type
            
        Returns:
            List of related data IDs
        """
        try:
            with self._lock:
                if data_id not in self._data or data_id not in self._relationships:
                    return []
                
                relationships = self._relationships[data_id]
                
                if relationship_type:
                    # Return specific relationship type
                    return relationships.get(relationship_type, [])
                else:
                    # Return all related IDs
                    related_ids = []
                    for rel_type, ids in relationships.items():
                        related_ids.extend(ids)
                    return related_ids
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'data_id': data_id},
                subsystem='data_pipeline',
                component='InMemoryDataRegistry'
            )
            return []
    
    def clean_expired_data(self) -> int:
        """
        Remove expired data
        
        Returns:
            Number of data entries removed
        """
        try:
            with self._lock:
                now = datetime.now()
                max_age = timedelta(seconds=self.parameters.get('max_data_age', 86400))
                
                to_remove = []
                for data_id, metadata in self._metadata.items():
                    # Check for explicit expiration
                    expiration_time = metadata.get('expiration_time')
                    if expiration_time and expiration_time <= now:
                        to_remove.append(data_id)
                    # Check for max age
                    elif 'creation_time' in metadata and (now - metadata['creation_time']) > max_age:
                        to_remove.append(data_id)
                
                # Remove data entries
                for data_id in to_remove:
                    self.remove_data(data_id)
                
                return len(to_remove)
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'operation': 'clean_expired_data'},
                subsystem='data_pipeline',
                component='InMemoryDataRegistry'
            )
            return 0
    
    def _update_indices_for_data(self, data_id: str, metadata: Dict[str, Any]) -> None:
        """
        Update all indices for a data entry
        
        Args:
            data_id: Data ID
            metadata: Data metadata
        """
        # Type index
        data_type = metadata.get('data_type')
        if data_type:
            if data_type not in self._type_index:
                self._type_index[data_type] = set()
            self._type_index[data_type].add(data_id)
        
        # Source index
        source = metadata.get('source')
        if source:
            if source not in self._source_index:
                self._source_index[source] = set()
            self._source_index[source].add(data_id)
        
        # Tag index
        tags = metadata.get('tags', [])
        for tag in tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(data_id)
        
        # Pair index
        pair = metadata.get('pair')
        if pair:
            if pair not in self._pair_index:
                self._pair_index[pair] = set()
            self._pair_index[pair].add(data_id)
        
        # Time index
        timeframe = metadata.get('timeframe')
        if timeframe:
            if timeframe not in self._time_index:
                self._time_index[timeframe] = set()
            self._time_index[timeframe].add(data_id)
    
    def _remove_from_indices(self, data_id: str, metadata: Dict[str, Any]) -> None:
        """
        Remove a data entry from all indices
        
        Args:
            data_id: Data ID
            metadata: Data metadata
        """
        # Type index
        data_type = metadata.get('data_type')
        if data_type and data_type in self._type_index:
            self._type_index[data_type].discard(data_id)
        
        # Source index
        source = metadata.get('source')
        if source and source in self._source_index:
            self._source_index[source].discard(data_id)
        
        # Tag index
        tags = metadata.get('tags', [])
        for tag in tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(data_id)
        
        # Pair index
        pair = metadata.get('pair')
        if pair and pair in self._pair_index:
            self._pair_index[pair].discard(data_id)
        
        # Time index
        timeframe = metadata.get('timeframe')
        if timeframe and timeframe in self._time_index:
            self._time_index[timeframe].discard(data_id)
    
    def _remove_relationships(self, data_id: str) -> None:
        """
        Remove all relationships for a data entry
        
        Args:
            data_id: Data ID
        """
        # Remove relationships where this data is the source
        if data_id in self._relationships:
            for rel_type, target_ids in self._relationships[data_id].items():
                # Remove inverse relationships
                inverse_type = f"inverse_{rel_type}" if not rel_type.startswith("inverse_") else rel_type[8:]
                for target_id in target_ids:
                    if target_id in self._relationships and inverse_type in self._relationships[target_id]:
                        if data_id in self._relationships[target_id][inverse_type]:
                            self._relationships[target_id][inverse_type].remove(data_id)
            
            # Remove this data's relationships
            del self._relationships[data_id]
        
        # Remove relationships where this data is the target
        for source_id, rel_dict in self._relationships.items():
            for rel_type, target_ids in list(rel_dict.items()):
                if data_id in target_ids:
                    target_ids.remove(data_id)
    
    def _remove_oldest_entries(self, count: int) -> None:
        """
        Remove the oldest data entries
        
        Args:
            count: Number of entries to remove
        """
        if not self._metadata:
            return
        
        # Sort entries by creation time
        sorted_entries = sorted(
            self._metadata.items(),
            key=lambda item: item[1].get('creation_time', datetime.min)
        )
        
        # Remove oldest entries
        for i in range(min(count, len(sorted_entries))):
            data_id, _ = sorted_entries[i]
            self.remove_data(data_id)
    
    def __del__(self):
        """Cleanup when registry is destroyed"""
        if self._auto_clean_thread and self._auto_clean_thread.is_alive():
            self._stop_auto_clean.set()
            self._auto_clean_thread.join(timeout=1.0)