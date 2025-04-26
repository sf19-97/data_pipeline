"""
Gap Detector - Validator for detecting time gaps in data

This component detects and reports gaps in time series data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import timedelta

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.data_pipeline.base import BaseValidator

@implements("DataValidator")
class GapDetector(BaseValidator):
    """
    Detector for time gaps in time series data
    
    Parameters:
    -----------
    max_gap_tolerance : Dict[str, float]
        Maximum tolerable gap size by timeframe in hours (default: {'1H': 1.5, 'D': 36.0})
    min_gap_size : Dict[str, float]
        Minimum gap size to report by timeframe in hours (default: {'1H': 1.0, 'D': 24.0})
    gap_fill_methods : List[str]
        Available methods for filling gaps (default: ['ffill', 'linear', 'nearest', 'cubic'])
    default_fill_method : str
        Default method for filling gaps (default: 'ffill')
    auto_fill_gaps : bool
        Whether to automatically fill gaps (default: False)
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'max_gap_tolerance': {'1H': 1.5, 'D': 36.0, '5min': 0.25},
            'min_gap_size': {'1H': 1.0, 'D': 24.0, '5min': 0.08},
            'gap_fill_methods': ['ffill', 'linear', 'nearest', 'cubic'],
            'default_fill_method': 'ffill',
            'auto_fill_gaps': False
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def validate(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data for time gaps
        
        Args:
            data: Input data (Dict[str, pd.DataFrame] or pd.DataFrame)
            context: Pipeline context
            
        Returns:
            Validation results
        """
        try:
            results = {
                'status': 'pass',
                'issues': [],
                'gaps': {}
            }
            
            # Handle different data types
            if isinstance(data, dict) and all(isinstance(df, pd.DataFrame) for df in data.values()):
                # Dictionary of DataFrames
                for key, df in data.items():
                    gap_results = self._find_gaps(df, context, key)
                    results['gaps'][key] = gap_results
                    
                    # Update overall status based on gap results
                    if gap_results['status'] == 'fail' and results['status'] != 'fail':
                        results['status'] = 'fail'
                        results['issues'].append(f"Critical gaps found in {key}")
                    elif gap_results['status'] == 'warning' and results['status'] == 'pass':
                        results['status'] = 'warning'
                        results['issues'].append(f"Gaps found in {key}")
                
                # Auto-fill gaps if configured
                if self.parameters.get('auto_fill_gaps', False) and results['status'] != 'pass':
                    self._fill_gaps(data, results, context)
            
            elif isinstance(data, pd.DataFrame):
                # Single DataFrame
                gap_results = self._find_gaps(data, context)
                results['gaps']['data'] = gap_results
                
                # Update overall status based on gap results
                if gap_results['status'] == 'fail':
                    results['status'] = 'fail'
                    results['issues'].append("Critical gaps found in data")
                elif gap_results['status'] == 'warning':
                    results['status'] = 'warning'
                    results['issues'].append("Gaps found in data")
                
                # Auto-fill gaps if configured
                if self.parameters.get('auto_fill_gaps', False) and results['status'] != 'pass':
                    self._fill_gaps({'data': data}, results, context)
                    
            else:
                results['status'] = 'pass'
                results['issues'].append("No time series data to check for gaps")
            
            return results
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={},
                subsystem='data_pipeline',
                component='GapDetector'
            )
            return {'status': 'error', 'issues': [f"Error detecting gaps: {str(e)}"]}
    
    def _find_gaps(self, df: pd.DataFrame, context: Dict[str, Any], 
                 key: Optional[str] = None) -> Dict[str, Any]:
        """Find gaps in a DataFrame
        
        Args:
            df: DataFrame to check
            context: Pipeline context
            key: Optional key for the DataFrame (for logging)
            
        Returns:
            Gap detection results
        """
        results = {
            'status': 'pass',
            'gap_count': 0,
            'gaps': [],
            'largest_gap': None,
            'total_missing': 0
        }
        
        # Skip if DataFrame is empty or not time-indexed
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return results
        
        # Determine timeframe
        timeframe = context.get('timeframe', self._infer_timeframe(df))
        
        # Get gap tolerances for this timeframe
        max_gap_tolerance = self._get_tolerance_for_timeframe(timeframe, 'max_gap_tolerance')
        min_gap_size = self._get_tolerance_for_timeframe(timeframe, 'min_gap_size')
        
        # Calculate time differences
        time_diffs = df.index.to_series().diff().dropna()
        
        # Convert to hours for consistent comparison
        time_diffs_hours = time_diffs.dt.total_seconds() / 3600
        
        # Find gaps
        gaps = time_diffs_hours[time_diffs_hours > min_gap_size]
        
        if not gaps.empty:
            # Store gap information
            results['gap_count'] = len(gaps)
            largest_gap = gaps.max()
            results['largest_gap'] = float(largest_gap)
            
            # Calculate total missing points
            expected_interval = self._get_expected_interval(timeframe)
            if expected_interval:
                # Estimate missing points based on expected interval
                expected_interval_hours = expected_interval.total_seconds() / 3600
                missing_points = sum(int(gap / expected_interval_hours) - 1 for gap in gaps)
                results['total_missing'] = missing_points
            
            # Collect detailed gap information
            for i, (idx, gap) in enumerate(gaps.items()):
                # Limit to top 10 gaps for performance
                if i >= 10:
                    break
                    
                gap_info = {
                    'start': idx.strftime('%Y-%m-%d %H:%M:%S'),
                    'end': (idx + timedelta(hours=float(gap))).strftime('%Y-%m-%d %H:%M:%S'),
                    'size_hours': float(gap),
                    'critical': gap > max_gap_tolerance
                }
                results['gaps'].append(gap_info)
            
            # Set status based on gap sizes
            critical_gaps = [g for g in results['gaps'] if g['critical']]
            if critical_gaps:
                results['status'] = 'fail'
            else:
                results['status'] = 'warning'
                
            # Log gap information
            log_msg = f"Found {results['gap_count']} gaps"
            if key:
                log_msg += f" in {key}"
            log_msg += f", largest gap: {results['largest_gap']:.2f} hours"
            self.logger.info(log_msg)
        
        return results
    
    def _fill_gaps(self, data: Dict[str, pd.DataFrame], results: Dict[str, Any], 
                  context: Dict[str, Any]) -> None:
        """Fill gaps in the data
        
        Args:
            data: Dictionary of DataFrames
            results: Gap detection results
            context: Pipeline context
        """
        fill_method = context.get('gap_fill_method') or self.parameters.get('default_fill_method')
        
        for key, df in data.items():
            if key in results['gaps'] and results['gaps'][key]['gap_count'] > 0:
                try:
                    # Get expected interval
                    timeframe = context.get('timeframe', self._infer_timeframe(df))
                    expected_interval = self._get_expected_interval(timeframe)
                    
                    if not expected_interval:
                        continue
                    
                    # Create a complete index
                    start = df.index.min()
                    end = df.index.max()
                    full_idx = pd.date_range(start=start, end=end, freq=self._timeframe_to_freq(timeframe))
                    
                    # Reindex with filling method
                    if fill_method == 'linear' or fill_method == 'cubic' or fill_method == 'nearest':
                        # Interpolation methods
                        data[key] = df.reindex(full_idx).interpolate(method=fill_method)
                    else:
                        # Forward/backward fill
                        data[key] = df.reindex(full_idx).fillna(method='ffill').fillna(method='bfill')
                    
                    self.logger.info(f"Filled {results['gaps'][key]['gap_count']} gaps in {key}")
                    
                except Exception as e:
                    self.error_handler.handle_error(
                        e,
                        context={'key': key, 'fill_method': fill_method},
                        subsystem='data_pipeline',
                        component='GapDetector'
                    )
    
    def _infer_timeframe(self, df: pd.DataFrame) -> str:
        """Infer timeframe from DataFrame
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Inferred timeframe string
        """
        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
            return 'D'  # Default to daily
        
        # Calculate median time difference
        time_diffs = df.index.to_series().diff().dropna()
        median_diff = time_diffs.median()
        seconds = median_diff.total_seconds()
        
        # Map to common timeframes
        if seconds <= 60:
            return '1min'
        elif seconds <= 300:
            return '5min'
        elif seconds <= 900:
            return '15min'
        elif seconds <= 1800:
            return '30min'
        elif seconds <= 3600:
            return '1H'
        elif seconds <= 14400:
            return '4H'
        elif seconds <= 86400:
            return 'D'
        else:
            return 'W'
    
    def _get_tolerance_for_timeframe(self, timeframe: str, tolerance_type: str) -> float:
        """Get tolerance value for a specific timeframe
        
        Args:
            timeframe: Timeframe string
            tolerance_type: Type of tolerance ('max_gap_tolerance' or 'min_gap_size')
            
        Returns:
            Tolerance value in hours
        """
        tolerances = self.parameters.get(tolerance_type, {})
        
        # Use exact match if available
        if timeframe in tolerances:
            return tolerances[timeframe]
        
        # Default mappings if exact match not found
        default_mapping = {
            'min_gap_size': {
                '1min': 0.03,  # ~2 minutes
                '5min': 0.08,  # ~5 minutes
                '15min': 0.25,  # 15 minutes
                '30min': 0.5,   # 30 minutes
                '1H': 1.0,     # 1 hour
                '4H': 4.0,     # 4 hours
                'D': 24.0,     # 1 day
                'W': 168.0     # 1 week
            },
            'max_gap_tolerance': {
                '1min': 0.1,   # 6 minutes
                '5min': 0.25,  # 15 minutes
                '15min': 0.5,  # 30 minutes
                '30min': 1.0,  # 1 hour
                '1H': 1.5,     # 1.5 hours
                '4H': 6.0,     # 6 hours
                'D': 36.0,     # 1.5 days
                'W': 240.0     # 10 days
            }
        }
        
        if timeframe in default_mapping.get(tolerance_type, {}):
            return default_mapping[tolerance_type][timeframe]
        
        # Final fallback
        if tolerance_type == 'min_gap_size':
            return 1.0  # 1 hour default
        else:
            return 24.0  # 1 day default
    
    def _get_expected_interval(self, timeframe: str) -> Optional[timedelta]:
        """Get expected interval for a timeframe
        
        Args:
            timeframe: Timeframe string
            
        Returns:
            Expected interval as timedelta
        """
        timeframe_map = {
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '30min': timedelta(minutes=30),
            '1H': timedelta(hours=1),
            '4H': timedelta(hours=4),
            'D': timedelta(days=1),
            'W': timedelta(weeks=1)
        }
        
        return timeframe_map.get(timeframe)
    
    def _timeframe_to_freq(self, timeframe: str) -> str:
        """Convert timeframe to pandas frequency string
        
        Args:
            timeframe: Timeframe string
            
        Returns:
            Pandas frequency string
        """
        timeframe_map = {
            '1min': 'min',
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1H': 'H',
            '4H': '4H',
            'D': 'D',
            'W': 'W'
        }
        
        return timeframe_map.get(timeframe, 'D')