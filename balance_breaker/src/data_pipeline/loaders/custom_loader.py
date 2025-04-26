"""
Custom Loader - Loader for custom data sources

This component loads data from custom sources with flexible options.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable
import glob
import json
import csv
import io

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.data_pipeline.base import BaseLoader

@implements("DataLoader")
class CustomLoader(BaseLoader):
    """
    Loader for custom data sources with flexible configuration
    
    Parameters:
    -----------
    data_directory : str
        Directory containing data files (default: None)
    file_pattern : str
        Glob pattern for file matching (default: "*.csv")
    parse_dates : Union[bool, List[str]]
        Columns to parse as dates (default: True)
    index_col : Optional[Union[str, int]]
        Column to use as index (default: 0)
    converters : Dict
        Dict of functions for converting values in certain columns (default: None)
    custom_parser : Optional[Callable]
        Custom parsing function (default: None)
    encoding : str
        File encoding (default: "utf-8")
    delimiter : str
        File delimiter for CSV files (default: ",")
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'data_directory': None,
            'file_pattern': "*.csv",
            'parse_dates': True,
            'index_col': 0,
            'converters': None,
            'custom_parser': None,
            'encoding': "utf-8",
            'delimiter': ","
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def load_data(self, context: Dict[str, Any]) -> Any:
        """Load data from custom sources
        
        Args:
            context: Pipeline context with parameters:
                - file_path: Path to specific file (optional)
                - data_sources: List of data sources to load (optional)
                - source_type: Type of source ('csv', 'json', 'custom') (optional)
                
        Returns:
            Loaded data (Dict[str, pd.DataFrame] or pd.DataFrame)
        """
        try:
            # Check for specific file path in context
            file_path = context.get('file_path')
            if file_path:
                return self._load_single_file(file_path, context)
            
            # Check for data sources in context
            data_sources = context.get('data_sources')
            if data_sources:
                return self._load_multiple_sources(data_sources, context)
            
            # Default to loading from directory with pattern
            return self._load_from_directory(context)
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={
                    'file_path': context.get('file_path'),
                    'data_sources': context.get('data_sources')
                },
                subsystem='data_pipeline',
                component='CustomLoader'
            )
            # Return empty result on error
            return {} if context.get('data_sources') or context.get('return_dict', False) else pd.DataFrame()
    
    def _load_single_file(self, file_path: str, context: Dict[str, Any]) -> pd.DataFrame:
        """Load data from a single file
        
        Args:
            file_path: Path to file
            context: Pipeline context
            
        Returns:
            DataFrame with loaded data
        """
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            self.error_handler.handle_error(
                FileNotFoundError(error_msg),
                context={'file_path': file_path},
                subsystem='data_pipeline',
                component='CustomLoader'
            )
            return pd.DataFrame()
        
        # Determine source type
        source_type = context.get('source_type') or self._guess_source_type(file_path)
        
        # Load based on source type
        if source_type == 'csv':
            return self._load_csv(file_path, context)
        elif source_type == 'json':
            return self._load_json(file_path, context)
        elif source_type == 'excel':
            return self._load_excel(file_path, context)
        elif source_type == 'custom':
            return self._load_custom(file_path, context)
        else:
            self.logger.warning(f"Unknown source type: {source_type}, trying CSV")
            return self._load_csv(file_path, context)
    
    def _load_multiple_sources(self, data_sources: List[Dict[str, Any]], 
                              context: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Load data from multiple sources
        
        Args:
            data_sources: List of data source configurations
            context: Pipeline context
            
        Returns:
            Dictionary of DataFrames with loaded data
        """
        result = {}
        
        for source in data_sources:
            try:
                source_name = source.get('name') or os.path.basename(source.get('file_path', ''))
                file_path = source.get('file_path')
                
                if file_path:
                    # Create source-specific context with source parameters
                    source_context = {**context, **source}
                    df = self._load_single_file(file_path, source_context)
                    
                    # Apply date filtering if specified
                    df = self._apply_date_filtering(df, context)
                    
                    result[source_name] = df
                    self.logger.info(f"Loaded source {source_name}: {len(df)} rows")
                else:
                    self.logger.warning(f"Missing file_path for source: {source_name}")
                    
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context={'source': source},
                    subsystem='data_pipeline',
                    component='CustomLoader'
                )
        
        return result
    
    def _load_from_directory(self, context: Dict[str, Any]) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Load data from directory using file pattern
        
        Args:
            context: Pipeline context
            
        Returns:
            Dictionary of DataFrames or single DataFrame
        """
        # Get directory and pattern
        directory = context.get('data_directory') or self.parameters.get('data_directory')
        if not directory or not os.path.exists(directory):
            error_msg = f"Directory not found: {directory}"
            self.error_handler.handle_error(
                FileNotFoundError(error_msg),
                context={'directory': directory},
                subsystem='data_pipeline',
                component='CustomLoader'
            )
            return {} if context.get('return_dict', False) else pd.DataFrame()
        
        # Find files matching pattern
        pattern = context.get('file_pattern') or self.parameters.get('file_pattern')
        file_paths = glob.glob(os.path.join(directory, pattern))
        
        if not file_paths:
            self.logger.warning(f"No files matching pattern {pattern} in {directory}")
            return {} if context.get('return_dict', False) else pd.DataFrame()
        
        # Load multiple files if return_dict is True
        if context.get('return_dict', False):
            result = {}
            for file_path in file_paths:
                try:
                    # Use filename as key
                    key = os.path.splitext(os.path.basename(file_path))[0]
                    result[key] = self._load_single_file(file_path, context)
                except Exception as e:
                    self.error_handler.handle_error(
                        e,
                        context={'file_path': file_path},
                        subsystem='data_pipeline',
                        component='CustomLoader'
                    )
            return result
        else:
            # Load just the first file
            return self._load_single_file(file_paths[0], context)
    
    def _load_csv(self, file_path: str, context: Dict[str, Any]) -> pd.DataFrame:
        """Load CSV file
        
        Args:
            file_path: Path to CSV file
            context: Pipeline context
            
        Returns:
            DataFrame with loaded data
        """
        # Get parameters
        index_col = context.get('index_col') or self.parameters.get('index_col')
        parse_dates = context.get('parse_dates') or self.parameters.get('parse_dates')
        encoding = context.get('encoding') or self.parameters.get('encoding')
        delimiter = context.get('delimiter') or self.parameters.get('delimiter')
        converters = context.get('converters') or self.parameters.get('converters')
        
        # Load CSV
        df = pd.read_csv(
            file_path,
            index_col=index_col,
            parse_dates=parse_dates,
            encoding=encoding,
            sep=delimiter,
            converters=converters
        )
        
        # Apply date filtering
        return self._apply_date_filtering(df, context)
    
    def _load_json(self, file_path: str, context: Dict[str, Any]) -> pd.DataFrame:
        """Load JSON file
        
        Args:
            file_path: Path to JSON file
            context: Pipeline context
            
        Returns:
            DataFrame with loaded data
        """
        # Get parameters
        orient = context.get('json_orient', 'records')
        
        # Load JSON
        df = pd.read_json(file_path, orient=orient)
        
        # Set index if specified
        index_col = context.get('index_col') or self.parameters.get('index_col')
        if index_col is not None and index_col in df.columns:
            df = df.set_index(index_col)
        
        # Convert date columns
        parse_dates = context.get('parse_dates') or self.parameters.get('parse_dates')
        if isinstance(parse_dates, list):
            for col in parse_dates:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        
        # Apply date filtering
        return self._apply_date_filtering(df, context)
    
    def _load_excel(self, file_path: str, context: Dict[str, Any]) -> pd.DataFrame:
        """Load Excel file
        
        Args:
            file_path: Path to Excel file
            context: Pipeline context
            
        Returns:
            DataFrame with loaded data
        """
        # Get parameters
        sheet_name = context.get('sheet_name', 0)
        index_col = context.get('index_col') or self.parameters.get('index_col')
        parse_dates = context.get('parse_dates') or self.parameters.get('parse_dates')
        
        # Load Excel
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            index_col=index_col,
            parse_dates=parse_dates
        )
        
        # Apply date filtering
        return self._apply_date_filtering(df, context)
    
    def _load_custom(self, file_path: str, context: Dict[str, Any]) -> pd.DataFrame:
        """Load data using custom parser
        
        Args:
            file_path: Path to file
            context: Pipeline context
            
        Returns:
            DataFrame with loaded data
        """
        # Get custom parser from parameters or context
        custom_parser = context.get('custom_parser') or self.parameters.get('custom_parser')
        
        if not custom_parser or not callable(custom_parser):
            error_msg = "Custom parser not provided or not callable"
            self.error_handler.handle_error(
                ValueError(error_msg),
                context={'file_path': file_path},
                subsystem='data_pipeline',
                component='CustomLoader'
            )
            return pd.DataFrame()
        
        # Call custom parser
        df = custom_parser(file_path, context)
        
        # Apply date filtering
        return self._apply_date_filtering(df, context)
    
    def _apply_date_filtering(self, df: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Apply date filtering to DataFrame
        
        Args:
            df: DataFrame to filter
            context: Pipeline context
            
        Returns:
            Filtered DataFrame
        """
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        # Apply date filtering
        start_date = context.get('start_date')
        end_date = context.get('end_date')
        
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        return df
    
    def _guess_source_type(self, file_path: str) -> str:
        """Guess the source type from file extension
        
        Args:
            file_path: Path to file
            
        Returns:
            Source type string
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.csv', '.txt']:
            return 'csv'
        elif file_ext in ['.json']:
            return 'json'
        elif file_ext in ['.xlsx', '.xls']:
            return 'excel'
        else:
            return 'csv'  # Default to CSV