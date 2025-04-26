"""
Base Classes for Modular Technical Indicators

This module provides the base classes for implementing modular indicators.
"""

import pandas as pd
from abc import abstractmethod
from typing import Dict, Any, List, Optional, Set, Type, ClassVar

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.data_pipeline.base import BaseIndicator


class ModularIndicator(BaseIndicator):
    """
    Base class for all modular indicators
    
    This class extends BaseIndicator and adds metadata about the indicator
    including its name, category, and required input fields.
    """
    
    # Class variables to be overridden by subclasses
    indicator_name: ClassVar[str] = "base"  # Name used in requests (e.g., "rsi", "macd")
    indicator_category: ClassVar[str] = "technical"  # Category (technical, economic, composite)
    required_columns: ClassVar[Set[str]] = {"close"}  # Minimum required input columns
    
    @property
    def name(self) -> str:
        """Return component name (override from PipelineComponent)"""
        return self.indicator_name
    
    @classmethod
    def get_indicator_name(cls) -> str:
        """Get the indicator name used in requests"""
        return cls.indicator_name
    
    @classmethod
    def get_indicator_category(cls) -> str:
        """Get the indicator category"""
        return cls.indicator_category
    
    @classmethod
    def get_required_columns(cls) -> Set[str]:
        """Get the required input columns"""
        return cls.required_columns
    
    @classmethod
    def can_calculate(cls, dataframe) -> bool:
        """Check if this indicator can be calculated with the given dataframe"""
        return all(col in dataframe.columns for col in cls.required_columns)
    
    @abstractmethod
    def calculate_indicator(self, df, **kwargs) -> Dict[str, Any]:
        """
        Calculate the indicator for the given dataframe
        
        Args:
            df: Input dataframe
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping new column names to their values
        """
        pass
    
    def calculate(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Calculate this indicator (implements IndicatorCalculator interface)
        
        Args:
            data: Input data (Dict or DataFrame)
            context: Pipeline context
            
        Returns:
            Updated data with indicator columns added
        """
        try:
            # Handle different input types
            if isinstance(data, dict) and 'price' in data:
                # Process price dictionary
                for pair, df in data['price'].items():
                    if self.can_calculate(df):
                        indicator_columns = self.calculate_indicator(df, **context)
                        # Add calculated columns to dataframe
                        for col_name, values in indicator_columns.items():
                            data['price'][pair][col_name] = values
                return data
                
            elif isinstance(data, dict) and all(isinstance(df, pd.DataFrame) for df in data.values()):
                # Process dictionary of dataframes
                for pair, df in data.items():
                    if self.can_calculate(df):
                        indicator_columns = self.calculate_indicator(df, **context)
                        # Add calculated columns to dataframe
                        for col_name, values in indicator_columns.items():
                            data[pair][col_name] = values
                return data
                
            else:
                # Process single dataframe
                if isinstance(data, pd.DataFrame) and self.can_calculate(data):
                    indicator_columns = self.calculate_indicator(data, **context)
                    # Add calculated columns to dataframe
                    result = data.copy()
                    for col_name, values in indicator_columns.items():
                        result[col_name] = values
                    return result
                
                # Return unchanged if can't process
                self.logger.warning(f"Indicator {self.indicator_name} can't process input data")
                return data
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'indicator': self.indicator_name},
                subsystem='data_pipeline',
                component=f'Indicator_{self.indicator_name}'
            )
            # Return input data on error
            return data


class IndicatorRegistry:
    """
    Registry for all available indicators
    
    This class maintains a registry of all indicator implementations,
    allowing discovery by name or category.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IndicatorRegistry, cls).__new__(cls)
            cls._instance.indicators = {}
            cls._instance.indicator_classes = set()
        return cls._instance
    
    def register_indicator(self, indicator_class: Type[ModularIndicator]) -> None:
        """
        Register an indicator class
        
        Args:
            indicator_class: Indicator class to register
        """
        name = indicator_class.get_indicator_name()
        if name in self.indicators:
            # Already registered
            return
            
        self.indicators[name] = indicator_class
        self.indicator_classes.add(indicator_class)
    
    def get_indicator_by_name(self, name: str) -> Optional[Type[ModularIndicator]]:
        """
        Get indicator class by name
        
        Args:
            name: Indicator name
            
        Returns:
            Indicator class or None if not found
        """
        return self.indicators.get(name)
    
    def get_indicators_by_category(self, category: str) -> List[Type[ModularIndicator]]:
        """
        Get all indicators in a category
        
        Args:
            category: Indicator category
            
        Returns:
            List of indicator classes
        """
        return [cls for cls in self.indicator_classes 
                if cls.get_indicator_category() == category]
    
    def list_all_indicators(self) -> List[str]:
        """
        Get a list of all registered indicator names
        
        Returns:
            List of indicator names
        """
        return list(self.indicators.keys())


# Create singleton instance
indicator_registry = IndicatorRegistry()


def register_indicator(cls):
    """
    Decorator to register an indicator class
    
    Usage:
        @register_indicator
        @implements("IndicatorCalculator")
        class RSIIndicator(ModularIndicator):
            indicator_name = "rsi"
            ...
    """
    indicator_registry.register_indicator(cls)
    return cls