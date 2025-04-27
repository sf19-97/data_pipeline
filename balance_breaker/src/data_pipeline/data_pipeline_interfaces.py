"""
Data Pipeline Interface Contracts

This module defines the core interfaces for the data pipeline subsystem.
These interfaces provide contracts that components must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from balance_breaker.src.core.interface_registry import interface

@interface
class PipelineComponent(ABC):
    """
    Base interface for all pipeline components
    
    All components in the data pipeline must implement this interface
    to ensure compatibility with the orchestrator.
    """
    
    @property
    @abstractmethod
    def component_type(self) -> str:
        """
        Return component type identifier
        
        Returns:
            String identifier for the component type
            (e.g., 'loader', 'processor', 'validator')
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
    
    @abstractmethod
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Process data according to component logic
        
        Args:
            data: Input data (can be None for initial components like loaders)
            context: Pipeline context information
            
        Returns:
            Processed data
        """
        pass


@interface
class DataLoader(PipelineComponent):
    """
    Interface for data loading components
    
    Loaders are responsible for loading data from various sources.
    """
    
    @abstractmethod
    def load_data(self, context: Dict[str, Any]) -> Any:
        """
        Load data from source
        
        Args:
            context: Pipeline context with loading parameters
            
        Returns:
            Loaded data
        """
        pass
    
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Process by loading data
        
        Default implementation that calls load_data
        
        Args:
            data: Input data (ignored for loaders)
            context: Pipeline context information
            
        Returns:
            Loaded data
        """
        return self.load_data(context)


@interface
class DataValidator(PipelineComponent):
    """
    Interface for data validation components
    
    Validators check data quality and correctness.
    """
    
    @abstractmethod
    def validate(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the data
        
        Args:
            data: Input data to validate
            context: Pipeline context
            
        Returns:
            Validation results
        """
        pass
    
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Process by validating data
        
        Default implementation that calls validate
        
        Args:
            data: Input data
            context: Pipeline context information
            
        Returns:
            Original data with validation results added to context
        """
        validation_results = self.validate(data, context)
        
        # Add validation results to context
        if 'validation' not in context:
            context['validation'] = {}
        
        context['validation'].update(validation_results)
        
        # Return original data
        return data


@interface
class DataProcessor(PipelineComponent):
    """
    Interface for data processing components
    
    Processors transform data in various ways.
    """
    
    @abstractmethod
    def process_data(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Process the data
        
        Args:
            data: Input data to process
            context: Pipeline context
            
        Returns:
            Processed data
        """
        pass
    
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Process data using process_data method
        
        Default implementation that calls process_data
        
        Args:
            data: Input data
            context: Pipeline context information
            
        Returns:
            Processed data
        """
        return self.process_data(data, context)


@interface
class DataAligner(PipelineComponent):
    """
    Interface for data alignment components
    
    Aligners synchronize multiple data series in time.
    """
    
    @abstractmethod
    def align(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Align data in time
        
        Args:
            data: Input data to align
            context: Pipeline context
            
        Returns:
            Aligned data
        """
        pass
    
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Process by aligning data
        
        Default implementation that calls align
        
        Args:
            data: Input data
            context: Pipeline context information
            
        Returns:
            Aligned data
        """
        return self.align(data, context)


@interface
class IndicatorCalculator(PipelineComponent):
    """
    Interface for indicator calculation components
    
    Indicator calculators generate indicators from price and other data.
    """
    
    @abstractmethod
    def calculate(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Calculate indicators
        
        Args:
            data: Input data to calculate indicators from
            context: Pipeline context
            
        Returns:
            Data with indicators added
        """
        pass
    
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Process by calculating indicators
        
        Default implementation that calls calculate
        
        Args:
            data: Input data
            context: Pipeline context information
            
        Returns:
            Data with indicators added
        """
        return self.calculate(data, context)


@interface
class DataSerializer(PipelineComponent):
    """
    Interface for data serialization components
    
    Serializers export or store data in various formats.
    """
    
    @abstractmethod
    def serialize(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Serialize data
        
        Args:
            data: Input data to serialize
            context: Pipeline context
            
        Returns:
            Serialization result
        """
        pass
    
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Process by serializing data
        
        Default implementation that calls serialize
        
        Args:
            data: Input data
            context: Pipeline context information
            
        Returns:
            Input data (unchanged) with serialization info in context
        """
        serialization_result = self.serialize(data, context)
        
        # Add serialization result to context
        if 'serialization' not in context:
            context['serialization'] = {}
        
        if isinstance(serialization_result, dict):
            context['serialization'].update(serialization_result)
        else:
            context['serialization']['result'] = serialization_result
        
        # Return original data
        return data


@interface
class DataRegistry(ABC):
    """
    Interface for data registry components
    
    Data registries manage data registration, lookup, and lifecycle.
    """
    
    @abstractmethod
    def register_data(self, data: Any, metadata: Dict[str, Any]) -> str:
        """
        Register data in the registry
        
        Args:
            data: Data to register
            metadata: Metadata for the data
            
        Returns:
            Data ID
        """
        pass
    
    @abstractmethod
    def get_data(self, data_id: str) -> Optional[Any]:
        """
        Get data by ID
        
        Args:
            data_id: ID of the data to retrieve
            
        Returns:
            Data if found, None otherwise
        """
        pass
    
    @abstractmethod
    def find_data(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find data matching criteria
        
        Args:
            criteria: Dictionary of search criteria
            
        Returns:
            List of data entries with metadata
        """
        pass
    
    @abstractmethod
    def update_data(self, data_id: str, data: Any) -> bool:
        """
        Update existing data
        
        Args:
            data_id: ID of the data to update
            data: New data
            
        Returns:
            True if update was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def remove_data(self, data_id: str) -> bool:
        """
        Remove data from the registry
        
        Args:
            data_id: ID of the data to remove
            
        Returns:
            True if removal was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def create_relationship(self, source_id: str, target_id: str, 
                          relationship_type: str) -> bool:
        """
        Create relationship between data products
        
        Args:
            source_id: Source data ID
            target_id: Target data ID
            relationship_type: Type of relationship
            
        Returns:
            True if relationship was created, False otherwise
        """
        pass
    
    @abstractmethod
    def get_related_data(self, data_id: str, 
                       relationship_type: Optional[str] = None) -> List[str]:
        """
        Get related data IDs
        
        Args:
            data_id: Data ID to find relations for
            relationship_type: Optional filter by relationship type
            
        Returns:
            List of related data IDs
        """
        pass
    
    @abstractmethod
    def clean_expired_data(self) -> int:
        """
        Remove expired data
        
        Returns:
            Number of data entries removed
        """
        pass