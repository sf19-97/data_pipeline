"""
Time Resampler - Resamples time series to different frequencies

This component resamples time series data to different timeframes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List, Optional

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.data_pipeline.base import BaseAligner

@implements("DataAligner")
class TimeResampler(BaseAligner):
    """
    Resamples time series data to different frequencies
    
    Parameters:
    -----------
    ohlc_columns : List[str]
        Column names for OHLC data (default: ['open', 'high', 'low', 'close'])
    default_method : str
        Default resampling method for non-OHLC data (default: 'last')
    volume_method : str
        Method for resampling volume data (default: 'sum')
    interpolate_method : str
        Method for interpolation when upsampling (default: 'time')
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'ohlc_columns': ['open', 'high', 'low', 'close'],
            'default_method': 'last',
            'volume_method': 'sum',
            'interpolate_method': 'time'
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def align_data(self, data: Any, context: Dict[str, Any]) -> Any:
        """Resample time series data to target frequency
        
        Args:
            data: Input data (Dict[str, pd.DataFrame] or pd.DataFrame)
            context: Pipeline context with parameters:
                - target_timeframe: Target timeframe (e.g., '1H', '4H', 'D')
                - resample_method: Method for resampling (optional)
                
        Returns:
            Resampled data
        """
        try:
            target_timeframe = context.get('target_timeframe')
            if not target_timeframe:
                self.logger.warning("No target timeframe specified for resampling")
                return data
                
            self.logger.info(f"Resampling data to {target_timeframe}")
            
            if isinstance(data, dict):
                # Process dictionary of dataframes
                result = {}
                for key, df in data.items():
                    try:
                        result[key] = self._resample_dataframe(df, target_timeframe, context)
                        self.logger.info(f"Resampled {key} to {target_timeframe}: {len(result[key])} rows")
                    except Exception as e:
                        self.error_handler.handle_error(
                            e,
                            context={'key': key, 'target_timeframe': target_timeframe},
                            subsystem='data_pipeline',
                            component='TimeResampler'
                        )
                        result[key] = df  # Keep original on error
                return result
                
            elif isinstance(data, pd.DataFrame):
                # Process single dataframe
                return self._resample_dataframe(data, target_timeframe, context)
                
            else:
                self.logger.warning(f"Unsupported data type for resampling: {type(data)}")
                return data
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'target_timeframe': context.get('target_timeframe')},
                subsystem='data_pipeline',
                component='TimeResampler'
            )
            raise
    
    def _resample_dataframe(self, df: pd.DataFrame, target_timeframe: str, 
                          context: Dict[str, Any]) -> pd.DataFrame:
        """Resample a single dataframe
        
        Args:
            df: Input dataframe
            target_timeframe: Target timeframe
            context: Pipeline context
            
        Returns:
            Resampled dataframe
        """
        # Make sure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Get OHLC columns from parameters
        ohlc_columns = self.parameters.get('ohlc_columns', ['open', 'high', 'low', 'close'])
        
        # Detect price data based on columns
        is_price_data = all(col in df.columns for col in ohlc_columns)
        
        if is_price_data:
            return self._resample_ohlc(df, target_timeframe, context)
        else:
            # For non-OHLC data, use standard resampling
            return self._resample_standard(df, target_timeframe, context)
    
    def _resample_ohlc(self, df: pd.DataFrame, target_timeframe: str, 
                      context: Dict[str, Any]) -> pd.DataFrame:
        """Resample OHLC price data
        
        Args:
            df: Price dataframe with OHLC columns
            target_timeframe: Target timeframe
            context: Pipeline context
            
        Returns:
            Resampled OHLC dataframe
        """
        # Get OHLC column names from parameters
        ohlc_columns = self.parameters.get('ohlc_columns', ['open', 'high', 'low', 'close'])
        
        # OHLC resampling rules
        resampler = df.resample(target_timeframe)
        
        # Build resampling dictionary for agg()
        agg_dict = {}
        
        # Check and add OHLC columns with appropriate functions
        if ohlc_columns[0] in df.columns:  # open
            agg_dict[ohlc_columns[0]] = 'first'
            
        if ohlc_columns[1] in df.columns:  # high
            agg_dict[ohlc_columns[1]] = 'max'
            
        if ohlc_columns[2] in df.columns:  # low
            agg_dict[ohlc_columns[2]] = 'min'
            
        if ohlc_columns[3] in df.columns:  # close
            agg_dict[ohlc_columns[3]] = 'last'
        
        # Handle volume column if present using volume_method parameter
        volume_method = self.parameters.get('volume_method', 'sum')
        if 'volume' in df.columns:
            agg_dict['volume'] = volume_method
        
        # Add any other columns using default method from parameters
        default_method = self.parameters.get('default_method', 'last')
        for col in df.columns:
            if col not in agg_dict:
                agg_dict[col] = default_method
        
        # Perform resampling with aggregation dictionary
        result = resampler.agg(agg_dict)
        
        return result
    
    def _resample_standard(self, df: pd.DataFrame, target_timeframe: str, 
                         context: Dict[str, Any]) -> pd.DataFrame:
        """Resample standard (non-OHLC) data
        
        Args:
            df: Non-OHLC dataframe
            target_timeframe: Target timeframe
            context: Pipeline context
            
        Returns:
            Resampled dataframe
        """
        # Determine appropriate method based on data and context
        method = context.get('resample_method', self.parameters.get('default_method', 'last'))
        
        if method == 'last':
            return df.resample(target_timeframe).last()
        elif method == 'mean':
            return df.resample(target_timeframe).mean()
        elif method == 'sum':
            return df.resample(target_timeframe).sum()
        elif method == 'interpolate':
            # Resample with last and then interpolate
            interpolate_method = self.parameters.get('interpolate_method', 'time')
            resampled = df.resample(target_timeframe).last()
            return resampled.interpolate(method=interpolate_method)
        else:
            # Default to last value
            return df.resample(target_timeframe).last()
    
    def downsample(self, data: pd.DataFrame, target_timeframe: str, method: Optional[str] = None) -> pd.DataFrame:
        """Convenience method for downsampling (e.g., 1H -> 4H)
        
        Args:
            data: Input dataframe
            target_timeframe: Target timeframe
            method: Resampling method (optional)
            
        Returns:
            Downsampled dataframe
        """
        context = {'resample_method': method or self.parameters.get('default_method', 'last')}
        return self._resample_dataframe(data, target_timeframe, context)
    
    def upsample(self, data: pd.DataFrame, target_timeframe: str, method: Optional[str] = None) -> pd.DataFrame:
        """Convenience method for upsampling (e.g., D -> 1H)
        
        Args:
            data: Input dataframe
            target_timeframe: Target timeframe
            method: Resampling method (optional)
            
        Returns:
            Upsampled dataframe
        """
        context = {'resample_method': method or 'interpolate'}
        return self._resample_dataframe(data, target_timeframe, context)