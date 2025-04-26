"""
Data Validator - Validator for price and macro data

This component validates data quality and reports issues.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.data_pipeline.base import BaseValidator

@implements("DataValidator")
class DataValidator(BaseValidator):
    """
    Validator for price and macro data
    
    Parameters:
    -----------
    required_price_columns : List[str]
        Required columns for price data (default: ['open', 'high', 'low', 'close'])
    warning_threshold : float
        Threshold for NaN values to trigger warning (as percentage) (default: 0.05)
    gap_warning_threshold : float
        Threshold for gaps to trigger warning (as percentage) (default: 0.01)
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'required_price_columns': ['open', 'high', 'low', 'close'],
            'warning_threshold': 0.05,  # 5% NaN values
            'gap_warning_threshold': 0.01  # 1% gaps
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def validate(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality
        
        Args:
            data: Input data (Dict[str, pd.DataFrame] for price, pd.DataFrame for macro)
            context: Pipeline context
            
        Returns:
            Validation results
        """
        try:
            data_type = context.get('data_type', 'price')
            
            if data_type == 'price':
                return self._validate_price_data(data, context)
            elif data_type == 'macro':
                return self._validate_macro_data(data, context)
            else:
                self.logger.warning(f"Unknown data type for validation: {data_type}")
                return {'status': 'unknown', 'issues': [f"Unknown data type: {data_type}"]}
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'data_type': context.get('data_type', 'unknown')},
                subsystem='data_pipeline',
                component='DataValidator'
            )
            return {'status': 'error', 'issues': [f"Validation error: {str(e)}"]}
    
    def _validate_price_data(self, data: Dict[str, pd.DataFrame], context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate price data
        
        Args:
            data: Dictionary of price DataFrames
            context: Pipeline context
            
        Returns:
            Validation results
        """
        results = {
            'status': 'pass',
            'issues': [],
            'pair_status': {}
        }
        
        # Check if data is empty
        if not data:
            results['status'] = 'fail'
            results['issues'].append("No price data found")
            return results
        
        required_columns = self.parameters.get('required_price_columns', ['open', 'high', 'low', 'close'])
        warning_threshold = self.parameters.get('warning_threshold', 0.05)
        gap_warning_threshold = self.parameters.get('gap_warning_threshold', 0.01)
        
        for pair, df in data.items():
            pair_results = {
                'status': 'pass',
                'issues': [],
                'missing_columns': [],
                'nan_counts': {},
                'row_count': len(df),
                'gap_count': 0
            }
            
            # Check for required columns
            for col in required_columns:
                if col not in df.columns:
                    pair_results['status'] = 'fail'
                    pair_results['issues'].append(f"Missing required column: {col}")
                    pair_results['missing_columns'].append(col)
            
            # If all required columns exist, check for NaN values
            if pair_results['status'] == 'pass':
                for col in df.columns:
                    nan_count = df[col].isna().sum()
                    if nan_count > 0:
                        pair_results['issues'].append(f"Column {col} has {nan_count} NaN values")
                        pair_results['nan_counts'][col] = nan_count
                        
                        # If NaN count is significant, mark as warning
                        if nan_count > len(df) * warning_threshold:
                            pair_results['status'] = 'warning'
                
                # Check for time gaps
                if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
                    # Calculate time differences
                    time_diffs = df.index.to_series().diff().dropna()
                    
                    # For hourly data, gaps are > 1 hour
                    if 'H1' in context.get('timeframe', 'H1'):
                        expected_diff = pd.Timedelta(hours=1)
                        gaps = time_diffs[time_diffs > expected_diff * 1.5]  # 1.5x tolerance
                    else:
                        # For daily data, gaps are > 1 day
                        expected_diff = pd.Timedelta(days=1)
                        gaps = time_diffs[time_diffs > expected_diff * 1.5]  # 1.5x tolerance
                    
                    gap_count = len(gaps)
                    if gap_count > 0:
                        pair_results['gap_count'] = gap_count
                        pair_results['issues'].append(f"Found {gap_count} time gaps in data")
                        
                        # If gap count is significant, mark as warning
                        if gap_count > len(df) * gap_warning_threshold:
                            pair_results['status'] = 'warning'
            
            # Add pair results
            results['pair_status'][pair] = pair_results
            
            # Update overall status
            if pair_results['status'] == 'fail' and results['status'] != 'fail':
                results['status'] = 'fail'
                results['issues'].append(f"Pair {pair} validation failed")
            elif pair_results['status'] == 'warning' and results['status'] == 'pass':
                results['status'] = 'warning'
                results['issues'].append(f"Pair {pair} has warnings")
        
        return results
    
    def _validate_macro_data(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate macro data
        
        Args:
            data: Macro data DataFrame
            context: Pipeline context
            
        Returns:
            Validation results
        """
        results = {
            'status': 'pass',
            'issues': [],
            'missing_columns': [],
            'nan_counts': {},
            'row_count': len(data) if isinstance(data, pd.DataFrame) else 0
        }
        
        # Check if data is empty
        if not isinstance(data, pd.DataFrame) or data.empty:
            results['status'] = 'fail'
            results['issues'].append("No macro data found or invalid format")
            return results
        
        warning_threshold = self.parameters.get('warning_threshold', 0.05)
        
        # Check for required indicators
        required_indicators = context.get('required_indicators', [])
        for indicator in required_indicators:
            if indicator not in data.columns:
                results['status'] = 'warning'
                results['issues'].append(f"Missing required indicator: {indicator}")
                results['missing_columns'].append(indicator)
        
        # Check for NaN values
        for col in data.columns:
            nan_count = data[col].isna().sum()
            if nan_count > 0:
                results['issues'].append(f"Indicator {col} has {nan_count} NaN values")
                results['nan_counts'][col] = nan_count
                
                # If NaN count is significant, mark as warning
                if nan_count > len(data) * warning_threshold:
                    results['status'] = 'warning'
        
        # Check for time consistency
        if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
            # Calculate time differences
            time_diffs = data.index.to_series().diff().dropna()
            
            # Most macro data is daily or monthly
            unique_diffs = time_diffs.value_counts().index
            
            # If multiple different time steps, might indicate inconsistency
            if len(unique_diffs) > 3:  # Allow for some variation
                results['status'] = 'warning'
                results['issues'].append(f"Inconsistent time intervals detected: {len(unique_diffs)} different intervals")
        
        return results