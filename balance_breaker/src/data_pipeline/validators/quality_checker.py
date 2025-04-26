"""
Quality Checker - Data quality verification component

This component checks the quality of price and macro data, identifying anomalies,
implausible values, and other data quality issues.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.data_pipeline.base import BaseValidator

@implements("DataValidator")
class QualityChecker(BaseValidator):
    """
    Validator for checking data quality metrics and issues
    
    Parameters:
    -----------
    price_limits : Dict[str, Dict[str, float]]
        Min/max limits for price data by column (default: auto-calculated)
    volatility_threshold : float
        Threshold for flagging excessive volatility (default: 4.0)
    sudden_change_threshold : float
        Threshold for flagging sudden changes as % (default: 10.0)
    zero_volume_warning : bool
        Whether to warn about zero volume periods (default: True)
    check_stationarity : bool
        Whether to perform stationarity tests (default: False)
    detect_outliers_std : float
        Standard deviation threshold for outlier detection (default: 3.0)
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'price_limits': None,  # Will be auto-calculated if None
            'volatility_threshold': 4.0,  # 4 standard deviations
            'sudden_change_threshold': 10.0,  # 10% change
            'zero_volume_warning': True,
            'check_stationarity': False,  # More complex check, disabled by default
            'detect_outliers_std': 3.0     # 3 standard deviations
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def validate(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality
        
        Args:
            data: Input data (Dict[str, pd.DataFrame] for price, pd.DataFrame for macro)
            context: Pipeline context
            
        Returns:
            Validation results with quality metrics and issues
        """
        try:
            data_type = context.get('data_type', 'price')
            
            if data_type == 'price':
                return self._validate_price_data(data, context)
            elif data_type == 'macro':
                return self._validate_macro_data(data, context)
            else:
                self.logger.warning(f"Unknown data type for quality check: {data_type}")
                return {'status': 'unknown', 'issues': [f"Unknown data type: {data_type}"]}
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'data_type': context.get('data_type', 'unknown')},
                subsystem='data_pipeline',
                component='QualityChecker'
            )
            return {'status': 'error', 'issues': [f"Quality check error: {str(e)}"]}
    
    def _validate_price_data(self, data: Dict[str, pd.DataFrame], context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate price data quality
        
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
        
        # Process each pair
        for pair, df in data.items():
            pair_results = {
                'status': 'pass',
                'issues': [],
                'quality_metrics': {},
                'outliers': {},
                'zero_volume_periods': 0,
                'anomalies': [],
            }
            
            # Skip if dataframe is empty
            if df.empty:
                pair_results['status'] = 'fail'
                pair_results['issues'].append("Empty dataframe")
                results['pair_status'][pair] = pair_results
                continue
            
            # 1. Check price ranges and implausible values
            if 'open' in df.columns and 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
                # Check for price integrity (high >= low, etc.)
                invalid_ohlc = (
                    (df['high'] < df['low']) | 
                    (df['open'] > df['high']) | 
                    (df['open'] < df['low']) | 
                    (df['close'] > df['high']) | 
                    (df['close'] < df['low'])
                )
                
                invalid_count = invalid_ohlc.sum()
                if invalid_count > 0:
                    pair_results['status'] = 'warning' if invalid_count < 5 else 'fail'
                    pair_results['issues'].append(f"Found {invalid_count} bars with invalid OHLC relationships")
                    pair_results['quality_metrics']['invalid_ohlc_count'] = invalid_count
                
                # Get price limits from parameters or auto-calculate
                price_limits = self.parameters.get('price_limits')
                if price_limits is None or pair not in price_limits:
                    # Auto-calculate reasonable limits based on data
                    median_price = df['close'].median()
                    price_range = df['high'].max() - df['low'].min()
                    
                    # Set limits at ±50% from median, or wider if price range demands it
                    min_limit = min(df['low'].min() * 0.9, median_price * 0.5)
                    max_limit = max(df['high'].max() * 1.1, median_price * 1.5)
                    
                    # Ensure limits are reasonable
                    if min_limit <= 0:
                        min_limit = df['low'].min() * 0.5
                    
                    limits = {
                        'low_min': min_limit,
                        'high_max': max_limit
                    }
                else:
                    limits = price_limits.get(pair, {})
                
                # Check for prices outside limits
                if 'low_min' in limits and (df['low'] < limits['low_min']).any():
                    below_min = (df['low'] < limits['low_min']).sum()
                    pair_results['issues'].append(f"Found {below_min} bars with price below min limit ({limits['low_min']})")
                    pair_results['outliers']['below_min'] = below_min
                    pair_results['status'] = 'warning'
                
                if 'high_max' in limits and (df['high'] > limits['high_max']).any():
                    above_max = (df['high'] > limits['high_max']).sum()
                    pair_results['issues'].append(f"Found {above_max} bars with price above max limit ({limits['high_max']})")
                    pair_results['outliers']['above_max'] = above_max
                    pair_results['status'] = 'warning'
            
            # 2. Check for price gaps and volatility
            if 'close' in df.columns:
                returns = df['close'].pct_change()
                volatility = returns.std()
                pair_results['quality_metrics']['volatility'] = volatility
                
                # Check for excessive volatility
                volatility_threshold = self.parameters.get('volatility_threshold', 4.0)
                sudden_change_threshold = self.parameters.get('sudden_change_threshold', 10.0) / 100.0
                
                # Check for sudden changes
                big_moves = (returns.abs() > sudden_change_threshold)
                big_move_count = big_moves.sum()
                
                if big_move_count > 0:
                    pair_results['issues'].append(f"Found {big_move_count} instances of sudden price changes > {sudden_change_threshold*100}%")
                    pair_results['quality_metrics']['big_move_count'] = big_move_count
                    
                    # Mark as warning if there are multiple big moves
                    if big_move_count > 2:
                        pair_results['status'] = 'warning'
                
                # Detect statistical outliers
                std_dev = returns.std()
                outlier_threshold = std_dev * self.parameters.get('detect_outliers_std', 3.0)
                outliers = (returns.abs() > outlier_threshold)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    pair_results['issues'].append(f"Found {outlier_count} statistical outliers in returns")
                    pair_results['outliers']['return_outliers'] = outlier_count
            
            # 3. Check for zero volume periods
            if 'volume' in df.columns and self.parameters.get('zero_volume_warning', True):
                zero_volume = (df['volume'] == 0)
                zero_volume_count = zero_volume.sum()
                
                if zero_volume_count > 0:
                    pair_results['issues'].append(f"Found {zero_volume_count} periods with zero volume")
                    pair_results['zero_volume_periods'] = zero_volume_count
                    
                    # Mark as warning if significant portion has zero volume
                    if zero_volume_count > len(df) * 0.05:  # More than 5%
                        pair_results['status'] = 'warning'
            
            # 4. Check for consistency in data frequency
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 2:
                time_diffs = df.index.to_series().diff().dropna()
                unique_diffs = time_diffs.value_counts()
                
                # If more than 3 different time intervals, might indicate inconsistency
                if len(unique_diffs) > 3:
                    pair_results['issues'].append(f"Inconsistent time intervals: {len(unique_diffs)} different intervals detected")
                    pair_results['quality_metrics']['time_diff_count'] = len(unique_diffs)
                    pair_results['status'] = 'warning'
            
            # 5. Check stationarity if enabled
            if self.parameters.get('check_stationarity', False) and 'close' in df.columns and len(df) > 30:
                try:
                    # Simple stationarity check using Augmented Dickey-Fuller test
                    from statsmodels.tsa.stattools import adfuller
                    
                    # Check stationarity of returns rather than prices
                    returns = df['close'].pct_change().dropna()
                    if len(returns) > 30:  # Need sufficient data
                        adf_result = adfuller(returns)
                        p_value = adf_result[1]
                        
                        pair_results['quality_metrics']['adf_p_value'] = p_value
                        
                        # Non-stationary returns is a serious issue
                        if p_value > 0.05:
                            pair_results['issues'].append(f"Returns may not be stationary (p-value: {p_value:.4f})")
                            pair_results['status'] = 'warning'
                except ImportError:
                    # statsmodels not available
                    self.logger.debug("statsmodels not available, skipping stationarity check")
                except Exception as e:
                    self.logger.debug(f"Stationarity check failed: {str(e)}")
            
            # Add pair results
            results['pair_status'][pair] = pair_results
            
            # Update overall status
            if pair_results['status'] == 'fail' and results['status'] != 'fail':
                results['status'] = 'fail'
                results['issues'].append(f"Pair {pair} quality check failed")
            elif pair_results['status'] == 'warning' and results['status'] == 'pass':
                results['status'] = 'warning'
                results['issues'].append(f"Pair {pair} has quality warnings")
        
        return results
    
    def _validate_macro_data(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate macro data quality
        
        Args:
            data: Macro data DataFrame
            context: Pipeline context
            
        Returns:
            Validation results
        """
        results = {
            'status': 'pass',
            'issues': [],
            'quality_metrics': {},
            'outliers': {},
            'anomalies': []
        }
        
        # Check if data is empty
        if not isinstance(data, pd.DataFrame) or data.empty:
            results['status'] = 'fail'
            results['issues'].append("No macro data found or invalid format")
            return results
        
        # 1. Check for excessive missing values
        missing_by_col = data.isna().sum()
        total_missing = missing_by_col.sum()
        
        if total_missing > 0:
            results['quality_metrics']['missing_values'] = total_missing
            results['quality_metrics']['missing_pct'] = total_missing / (len(data) * len(data.columns))
            
            # If more than 10% missing overall, mark as warning
            if results['quality_metrics']['missing_pct'] > 0.1:
                results['status'] = 'warning'
                results['issues'].append(f"High proportion of missing values: {results['quality_metrics']['missing_pct']:.1%}")
            
            # Find columns with excessive missing values
            bad_cols = missing_by_col[missing_by_col > len(data) * 0.2]  # More than 20% missing
            if len(bad_cols) > 0:
                results['issues'].append(f"Found {len(bad_cols)} columns with >20% missing values")
                results['quality_metrics']['high_missing_cols'] = len(bad_cols)
        
        # 2. Check for implausible values and outliers in each column
        outlier_cols = 0
        
        for col in data.columns:
            series = data[col]
            
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(series):
                continue
            
            # Skip columns with too many missing values
            if series.isna().sum() > len(series) * 0.5:
                continue
            
            # Calculate z-scores for outlier detection
            mean = series.mean()
            std = series.std()
            
            if std == 0:  # Skip constant columns
                continue
            
            # Detect outliers using z-score approach
            z_scores = (series - mean) / std
            outlier_threshold = self.parameters.get('detect_outliers_std', 3.0)
            outliers = (abs(z_scores) > outlier_threshold)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                results['outliers'][col] = outlier_count
                
                # If significant outliers, mark for attention
                if outlier_count > len(series) * 0.02:  # More than 2% are outliers
                    outlier_cols += 1
        
        if outlier_cols > 0:
            results['issues'].append(f"Found {outlier_cols} columns with significant outliers")
            results['quality_metrics']['outlier_cols'] = outlier_cols
            
            if outlier_cols > len(data.columns) * 0.1:  # More than 10% of columns have outliers
                results['status'] = 'warning'
        
        # 3. Check for consistency in data frequency
        if isinstance(data.index, pd.DatetimeIndex) and len(data) > 2:
            time_diffs = data.index.to_series().diff().dropna()
            unique_diffs = time_diffs.value_counts()
            
            results['quality_metrics']['time_diff_types'] = len(unique_diffs)
            
            # If more than 3 different time intervals, might indicate inconsistency
            if len(unique_diffs) > 3:
                results['issues'].append(f"Inconsistent time intervals: {len(unique_diffs)} different intervals detected")
                
                # Only mark as warning if significant inconsistency
                if len(unique_diffs) > 5:
                    results['status'] = 'warning'
        
        # 4. Check for excessive volatility in key indicators
        volatility_threshold = self.parameters.get('volatility_threshold', 4.0)
        
        for col in data.columns:
            # Focus on important macro indicators
            if any(key in col.lower() for key in ['inflation', 'cpi', 'gdp', 'rate', 'yield']):
                series = data[col]
                
                # Skip non-numeric or sparse columns
                if not pd.api.types.is_numeric_dtype(series) or series.isna().sum() > len(series) * 0.5:
                    continue
                
                # Calculate changes
                changes = series.pct_change().dropna()
                
                if len(changes) < 5:  # Skip if too few observations
                    continue
                
                # Check for sudden large changes
                std_dev = changes.std()
                large_changes = (abs(changes) > std_dev * volatility_threshold)
                large_change_count = large_changes.sum()
                
                if large_change_count > 0:
                    results['anomalies'].append({
                        'column': col,
                        'large_changes': large_change_count,
                        'max_change': changes.abs().max()
                    })
                    
                    if large_change_count > 2:  # Multiple large changes
                        results['issues'].append(f"Column {col} shows unusual volatility with {large_change_count} large changes")
                        
                        if results['status'] == 'pass':
                            results['status'] = 'warning'
        
        return results