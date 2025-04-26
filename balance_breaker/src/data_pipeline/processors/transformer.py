"""
Data Transformer - Transforms data with various operations

This component applies transformations to data formats or values.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.data_pipeline.base import BaseProcessor

@implements("DataProcessor")
class DataTransformer(BaseProcessor):
    """
    Transformer for data conversion operations
    
    Parameters:
    -----------
    transform_method : str
        Method for transformation ('log', 'sqrt', 'diff') (default: 'log')
    columns_to_transform : List[str]
        Specific columns to transform (default: None, transforms all numeric)
    add_original : bool
        Whether to keep original columns (default: True)
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'transform_method': 'log',
            'columns_to_transform': None,
            'add_original': True
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def process_data(self, data: Any, context: Dict[str, Any]) -> Any:
        """Apply transformations to data
        
        Args:
            data: Input data (DataFrame or Dict of DataFrames)
            context: Pipeline context
            
        Returns:
            Transformed data
        """
        try:
            # Handle dictionary of dataframes (e.g., price data by pair)
            if isinstance(data, dict) and all(isinstance(df, pd.DataFrame) for df in data.values()):
                result = {}
                for key, df in data.items():
                    self.logger.info(f"Transforming data for {key}")
                    result[key] = self._transform_dataframe(df, context)
                return result
            
            # Handle single dataframe
            elif isinstance(data, pd.DataFrame):
                self.logger.info("Transforming dataframe")
                return self._transform_dataframe(data, context)
            
            # Return unchanged for unsupported types
            else:
                self.logger.warning(f"Unsupported data type for transformation: {type(data)}")
                return data
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'transform_method': self.parameters.get('transform_method')},
                subsystem='data_pipeline',
                component='DataTransformer'
            )
            return data
    
    def _transform_dataframe(self, df: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Transform a single dataframe
        
        Args:
            df: DataFrame to transform
            context: Pipeline context
            
        Returns:
            Transformed DataFrame
        """
        # Create a copy to avoid modifying original
        result = df.copy()
        
        # Get parameters
        transform_method = self.parameters.get('transform_method', 'log')
        columns_to_transform = self.parameters.get('columns_to_transform')
        add_original = self.parameters.get('add_original', True)
        
        # Determine columns to transform
        if columns_to_transform:
            # Use specified columns that exist in the dataframe
            target_cols = [col for col in columns_to_transform if col in df.columns]
        else:
            # Use all numeric columns
            target_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            # Exclude typical non-transformable columns
            exclude_cols = ['volume', 'pip_factor']
            target_cols = [col for col in target_cols if col not in exclude_cols]
        
        # Skip if no columns to transform
        if not target_cols:
            self.logger.info("No suitable columns found for transformation")
            return result
            
        # Apply transformation based on method
        for col in target_cols:
            # Create the new column name
            new_col = f"{col}_{transform_method}" if add_original else col
            
            # Apply selected transformation
            if transform_method == 'log':
                # Handle negative values with offset if needed
                min_val = df[col].min()
                offset = abs(min_val) + 1 if min_val <= 0 else 0
                result[new_col] = np.log(df[col] + offset)
                
            elif transform_method == 'sqrt':
                # Handle negative values with offset if needed
                min_val = df[col].min()
                offset = abs(min_val) + 1 if min_val <= 0 else 0
                result[new_col] = np.sqrt(df[col] + offset)
                
            elif transform_method == 'diff':
                result[new_col] = df[col].diff().fillna(0)
                
            elif transform_method == 'pct_change':
                result[new_col] = df[col].pct_change().fillna(0)
                
            else:
                self.logger.warning(f"Unknown transformation method: {transform_method}")
        
        self.logger.info(f"Applied {transform_method} transformation to {len(target_cols)} columns")
        return result