"""
Base Components for Data Pipeline

This module defines the base classes for all data pipeline components.
These base classes implement the interface contracts from the core framework.
"""

import logging
from abc import abstractmethod
from typing import Dict, Any, Optional, List, Union

from balance_breaker.src.core.interface_registry import implements, interface
from balance_breaker.src.core.error_handling import ErrorHandler
from balance_breaker.src.core.parameter_manager import ParameterizedComponent
from balance_breaker.src.data_pipeline.data_pipeline_interfaces import (
    PipelineComponent, DataLoader, DataValidator, 
    DataProcessor, DataAligner, IndicatorCalculator, DataSerializer
)

class BasePipelineComponent(ParameterizedComponent, PipelineComponent):
    """Base class for all pipeline components
    
    Extends ParameterizedComponent from the core framework.
    Implements the PipelineComponent interface.
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
    
    @abstractmethod
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process data according to component logic
        
        Args:
            data: Input data (can be None for loaders)
            context: Pipeline context information
            
        Returns:
            Processed data
        """
        pass


@implements("DataLoader")
class BaseLoader(BasePipelineComponent):
    """Base class for data loaders
    
    Loaders are responsible for loading data from various sources.
    """
    
    @property
    def component_type(self) -> str:
        """Return component type identifier"""
        return 'loader'
    
    @abstractmethod
    def load_data(self, context: Dict[str, Any]) -> Any:
        """Load data from source
        
        Args:
            context: Pipeline context with loading parameters
            
        Returns:
            Loaded data
        """
        pass
    
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process by loading data
        
        Args:
            data: Input data (ignored for loaders)
            context: Pipeline context information
            
        Returns:
            Loaded data
        """
        return self.load_data(context)


@implements("DataValidator")
class BaseValidator(BasePipelineComponent):
    """Base class for data validators
    
    Validators check data quality and correctness.
    """
    
    @property
    def component_type(self) -> str:
        """Return component type identifier"""
        return 'validator'
    
    @abstractmethod
    def validate(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the data
        
        Args:
            data: Input data to validate
            context: Pipeline context
            
        Returns:
            Validation results
        """
        pass
    
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process by validating data
        
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


@implements("DataProcessor")
class BaseProcessor(BasePipelineComponent):
    """Base class for data processors
    
    Processors transform data in various ways.
    """
    
    @property
    def component_type(self) -> str:
        """Return component type identifier"""
        return 'processor'
    
    @abstractmethod
    def process_data(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process the data
        
        Args:
            data: Input data to process
            context: Pipeline context
            
        Returns:
            Processed data
        """
        pass
    
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process data using process_data method
        
        Args:
            data: Input data
            context: Pipeline context information
            
        Returns:
            Processed data
        """
        return self.process_data(data, context)


@implements("DataAligner")
class BaseAligner(BasePipelineComponent):
    """Base class for data aligners
    
    Aligners synchronize multiple data series in time.
    """
    
    @property
    def component_type(self) -> str:
        """Return component type identifier"""
        return 'aligner'
    
    @abstractmethod
    def align_data(self, data: Any, context: Dict[str, Any]) -> Any:
        """Align data in time
        
        Args:
            data: Input data to align
            context: Pipeline context
            
        Returns:
            Aligned data
        """
        pass
    
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process by aligning data
        
        Args:
            data: Input data
            context: Pipeline context information
            
        Returns:
            Aligned data
        """
        return self.align_data(data, context)


@implements("IndicatorCalculator")
class BaseIndicator(BasePipelineComponent):
    """Base class for indicator calculators
    
    Indicator calculators generate indicators from price and other data.
    """
    
    @property
    def component_type(self) -> str:
        """Return component type identifier"""
        return 'indicator'
    
    @abstractmethod
    def calculate(self, data: Any, context: Dict[str, Any]) -> Any:
        """Calculate indicators
        
        Args:
            data: Input data to calculate indicators from
            context: Pipeline context
            
        Returns:
            Data with indicators added
        """
        pass
    
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process by calculating indicators
        
        Args:
            data: Input data
            context: Pipeline context information
            
        Returns:
            Data with indicators added
        """
        return self.calculate(data, context)


@implements("DataSerializer")
class BaseSerializer(BasePipelineComponent):
    """Base class for data serializers
    
    Serializers export or store data in various formats.
    """
    
    @property
    def component_type(self) -> str:
        """Return component type identifier"""
        return 'serializer'
    
    @abstractmethod
    def serialize(self, data: Any, context: Dict[str, Any]) -> Any:
        """Serialize data
        
        Args:
            data: Input data to serialize
            context: Pipeline context
            
        Returns:
            Serialization result
        """
        pass
    
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process by serializing data
        
        Args:
            data: Input data
            context: Pipeline context information
            
        Returns:
            Original data (with serialization info in context)
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