"""
Data Exporter - Export data to various formats

This component exports data to various file formats and destinations.
"""

import os
import pandas as pd
import json
import pickle
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.data_pipeline.base import BaseSerializer

@implements("DataSerializer")
class DataExporter(BaseSerializer):
    """
    Exporter for saving data to various formats
    
    Parameters:
    -----------
    export_dir : str
        Directory to save exported files (default: 'exported_data')
    export_formats : List[str]
        List of formats to export to (default: ['csv'])
    timestamp_format : str
        Format for timestamp in filenames (default: '%Y%m%d_%H%M%S')
    include_timestamp : bool
        Whether to include timestamp in filenames (default: True)
    create_subfolders : bool
        Whether to create subfolders for different data types (default: True)
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'export_dir': 'exported_data',
            'export_formats': ['csv'],
            'timestamp_format': '%Y%m%d_%H%M%S',
            'include_timestamp': True,
            'create_subfolders': True
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def serialize(self, data: Any, context: Dict[str, Any]) -> Any:
        """Export data to specified formats
        
        Args:
            data: Input data to export
            context: Pipeline context with parameters:
                - export_formats: List of formats to export to (csv, excel, pickle, json)
                - export_dir: Directory to save exported files
                - export_prefix: Prefix for exported filenames
                
        Returns:
            Original data (export paths are added to context)
        """
        try:
            # Get export parameters from context or use defaults
            export_formats = context.get('export_formats', self.parameters.get('export_formats'))
            export_dir = context.get('export_dir', self.parameters.get('export_dir'))
            export_prefix = context.get('export_prefix', 'data_export')
            
            # Add timestamp if configured
            if self.parameters.get('include_timestamp', True):
                timestamp = datetime.now().strftime(self.parameters.get('timestamp_format'))
                export_prefix = f"{export_prefix}_{timestamp}"
            
            # Create export directory if it doesn't exist
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            
            # Initialize export results
            if 'export_results' not in context:
                context['export_results'] = {}
            
            # Determine data type and export accordingly
            if isinstance(data, pd.DataFrame):
                # Single DataFrame
                context['export_results'] = self._export_dataframe(
                    data, export_formats, export_dir, export_prefix, context
                )
                
            elif isinstance(data, dict) and self._contains_dataframes(data):
                # Dictionary of DataFrames or nested structure
                context['export_results'] = self._export_dataframe_dict(
                    data, export_formats, export_dir, export_prefix, context
                )
                
            else:
                # Other data types
                context['export_results'] = self._export_generic(
                    data, export_formats, export_dir, export_prefix, context
                )
            
            return data
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={
                    'export_formats': context.get('export_formats', self.parameters.get('export_formats')),
                    'export_dir': context.get('export_dir', self.parameters.get('export_dir'))
                },
                subsystem='data_pipeline',
                component='DataExporter'
            )
            # Return original data
            return data
    
    def _contains_dataframes(self, data_dict: Dict) -> bool:
        """Check if dictionary contains DataFrames
        
        Args:
            data_dict: Dictionary to check
            
        Returns:
            True if dictionary contains DataFrames, False otherwise
        """
        for value in data_dict.values():
            if isinstance(value, pd.DataFrame):
                return True
            elif isinstance(value, dict) and self._contains_dataframes(value):
                return True
        return False
    
    def _export_dataframe(self, df: pd.DataFrame, formats: List[str], 
                        export_dir: str, prefix: str, 
                        context: Dict[str, Any]) -> Dict[str, str]:
        """Export a single DataFrame to various formats
        
        Args:
            df: DataFrame to export
            formats: List of export formats
            export_dir: Directory to save exported files
            prefix: Filename prefix
            context: Pipeline context
            
        Returns:
            Dictionary of export paths by format
        """
        results = {}
        
        for fmt in formats:
            fmt = fmt.lower()
            
            try:
                if fmt == 'csv':
                    filename = f"{prefix}.csv"
                    filepath = os.path.join(export_dir, filename)
                    df.to_csv(filepath)
                    results['csv'] = filepath
                    self.logger.info(f"Exported DataFrame to CSV: {filepath}")
                    
                elif fmt == 'excel':
                    filename = f"{prefix}.xlsx"
                    filepath = os.path.join(export_dir, filename)
                    df.to_excel(filepath)
                    results['excel'] = filepath
                    self.logger.info(f"Exported DataFrame to Excel: {filepath}")
                    
                elif fmt == 'pickle':
                    filename = f"{prefix}.pkl"
                    filepath = os.path.join(export_dir, filename)
                    df.to_pickle(filepath)
                    results['pickle'] = filepath
                    self.logger.info(f"Exported DataFrame to pickle: {filepath}")
                    
                elif fmt == 'json':
                    filename = f"{prefix}.json"
                    filepath = os.path.join(export_dir, filename)
                    df.to_json(filepath)
                    results['json'] = filepath
                    self.logger.info(f"Exported DataFrame to JSON: {filepath}")
                    
                elif fmt == 'parquet':
                    filename = f"{prefix}.parquet"
                    filepath = os.path.join(export_dir, filename)
                    df.to_parquet(filepath)
                    results['parquet'] = filepath
                    self.logger.info(f"Exported DataFrame to Parquet: {filepath}")
                    
                else:
                    self.logger.warning(f"Unsupported export format: {fmt}")
                    
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context={'format': fmt, 'prefix': prefix},
                    subsystem='data_pipeline',
                    component='DataExporter'
                )
        
        return results
    
    def _export_dataframe_dict(self, data_dict: Dict, formats: List[str], 
                             export_dir: str, prefix: str, 
                             context: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Export a dictionary of DataFrames to various formats
        
        Args:
            data_dict: Dictionary of DataFrames
            formats: List of export formats
            export_dir: Directory to save exported files
            prefix: Filename prefix
            context: Pipeline context
            
        Returns:
            Dictionary of export results by key and format
        """
        results = {}
        
        # Create subfolders if configured
        create_subfolders = self.parameters.get('create_subfolders', True)
        
        for key, value in data_dict.items():
            # Create key-specific prefix
            key_prefix = f"{prefix}_{key}"
            
            # Create key-specific directory if configured
            key_dir = export_dir
            if create_subfolders:
                key_dir = os.path.join(export_dir, str(key))
                if not os.path.exists(key_dir):
                    os.makedirs(key_dir)
            
            if isinstance(value, pd.DataFrame):
                # Export DataFrame
                results[key] = self._export_dataframe(
                    value, formats, key_dir, key_prefix, context
                )
                
            elif isinstance(value, dict) and self._contains_dataframes(value):
                # Recursively export nested dictionary
                nested_results = self._export_dataframe_dict(
                    value, formats, key_dir, key_prefix, context
                )
                results[key] = nested_results
                
            else:
                # Skip non-DataFrame values
                pass
        
        return results
    
    def _export_generic(self, data: Any, formats: List[str], 
                      export_dir: str, prefix: str, 
                      context: Dict[str, Any]) -> Dict[str, str]:
        """Export generic data to various formats
        
        Args:
            data: Data to export
            formats: List of export formats
            export_dir: Directory to save exported files
            prefix: Filename prefix
            context: Pipeline context
            
        Returns:
            Dictionary of export paths by format
        """
        results = {}
        
        for fmt in formats:
            fmt = fmt.lower()
            
            try:
                if fmt == 'pickle':
                    filename = f"{prefix}.pkl"
                    filepath = os.path.join(export_dir, filename)
                    with open(filepath, 'wb') as f:
                        pickle.dump(data, f)
                    results['pickle'] = filepath
                    self.logger.info(f"Exported data to pickle: {filepath}")
                    
                elif fmt == 'json':
                    filename = f"{prefix}.json"
                    filepath = os.path.join(export_dir, filename)
                    
                    # Convert to JSON-serializable format
                    json_data = self._convert_to_json_serializable(data)
                    
                    with open(filepath, 'w') as f:
                        json.dump(json_data, f, indent=2)
                    results['json'] = filepath
                    self.logger.info(f"Exported data to JSON: {filepath}")
                    
                else:
                    self.logger.warning(f"Unsupported export format for generic data: {fmt}")
                    
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context={'format': fmt, 'prefix': prefix},
                    subsystem='data_pipeline',
                    component='DataExporter'
                )
        
        return results
    
    def _convert_to_json_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format
        
        Args:
            data: Data to convert
            
        Returns:
            JSON-serializable data
        """
        if isinstance(data, (str, int, float, bool, type(None))):
            return data
            
        elif isinstance(data, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in data]
            
        elif isinstance(data, dict):
            return {str(k): self._convert_to_json_serializable(v) for k, v in data.items()}
            
        elif isinstance(data, pd.DataFrame):
            return data.to_dict(orient='records')
            
        elif isinstance(data, pd.Series):
            return data.to_dict()
            
        elif hasattr(data, 'to_dict'):
            return data.to_dict()
            
        elif hasattr(data, '__dict__'):
            return self._convert_to_json_serializable(data.__dict__)
            
        else:
            return str(data)