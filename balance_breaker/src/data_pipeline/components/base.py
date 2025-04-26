# src/data_pipeline/components/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class PipelineComponent(ABC):
    """Base interface for all pipeline components"""
    
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
    
    @property
    @abstractmethod
    def component_type(self) -> str:
        """Return component type identifier"""
        pass
    
    @property
    def name(self) -> str:
        """Return component name"""
        return self.__class__.__name__