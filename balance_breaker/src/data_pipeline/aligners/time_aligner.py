"""
Time Aligner - Time-based data synchronization component

This component aligns multiple time series to a common timeline.
"""

import pandas as pd
from typing import Dict, Any, Union, List, Tuple

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.data_pipeline.base import BaseAligner

@implements("DataAligner")
class TimeAligner(BaseAligner):
    """
    Aligner for time-based data synchronization
    
    Parameters:
    -----------
    fill_method : str
        Method for filling missing values after alignment (default: 'ffill')
    align_to_highest_frequency : bool
        Whether to align to the highest frequency time series (default: True)
    handle_missing : str
        Method for handling missing values ('ffill', 'bfill', 'nearest', 'interpolate') (default: 'ffill')
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'fill_method': 'ffill',
            'align_to_highest_frequency': True,
            'handle_missing': 'ffill'
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def align_data(self, data: Any, context: Dict[str, Any]) -> Any:
        """Align price and macro data to common timeline
        
        Args:
            data: Input data (Dict[str, pd.DataFrame] for price, pd.DataFrame for macro)
            context: Pipeline context
            
        Returns:
            Dictionary with aligned price and macro data
        """
        try:
            data_type = context.get('data_type')
            
            # Determine what we're aligning
            if 'price_data' in context and 'macro_data' in context:
                # If we have both price and macro data in context, align them
                return self._align_price_with_macro(context['price_data'], context['macro_data'], context)
                
            elif isinstance(data, dict) and all(isinstance(df, pd.DataFrame) for df in data.values()):
                if data_type == 'price':
                    # If we're processing price data dict
                    price_data = data
                    macro_data = context.get('macro_data')
                    
                    if macro_data is not None and isinstance(macro_data, pd.DataFrame):
                        # If macro data is available, align price data with it
                        return self._align_price_with_macro(price_data, macro_data, context)
                    else:
                        # Otherwise just align price data timeframes with each other
                        return self._align_price_data(price_data, context)
                else:
                    # Generic dictionary of dataframes alignment
                    return self._align_multi_dataframes(data, context)
            
            elif isinstance(data, pd.DataFrame):
                if data_type == 'macro':
                    # Single macro dataframe, nothing to align with
                    return data
            
            # Return unchanged if no alignment needed or possible
            self.logger.warning("No suitable data for alignment found")
            return data
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'data_type': context.get('data_type', 'unknown')},
                subsystem='data_pipeline',
                component='TimeAligner'
            )
            raise
    
    def _align_price_with_macro(self, price_data: Dict[str, pd.DataFrame], 
                               macro_data: pd.DataFrame, 
                               context: Dict[str, Any]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Align price data with macro data
        
        Args:
            price_data: Dictionary of price dataframes by pair
            macro_data: Macro dataframe
            context: Pipeline context
            
        Returns:
            Dictionary with 'price' and 'macro' keys containing aligned data
        """
        self.logger.info("Aligning price data with macro data")
        
        # Create result dictionary
        result = {
            'price': {},
            'aligned_macro': {}
        }
        
        # Process each pair
        for pair, df in price_data.items():
            try:
                # Create aligned macro data for this pair
                aligned_macro = self._align_macro_to_price(macro_data, df, context)
                
                # Store results
                result['price'][pair] = df
                result['aligned_macro'][pair] = aligned_macro
                
                self.logger.info(f"Aligned macro data with {pair} price data: {len(aligned_macro)} rows")
                
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context={'pair': pair},
                    subsystem='data_pipeline',
                    component='TimeAligner'
                )
        
        return result
    
    def _align_macro_to_price(self, macro_df: pd.DataFrame, 
                             price_df: pd.DataFrame, 
                             context: Dict[str, Any]) -> pd.DataFrame:
        """Align macro data to price data timeline
        
        Args:
            macro_df: Macro dataframe
            price_df: Price dataframe
            context: Pipeline context
            
        Returns:
            Aligned macro dataframe
        """
        # Make sure indexes are datetime
        if not isinstance(macro_df.index, pd.DatetimeIndex):
            self.logger.warning("Macro data index is not DatetimeIndex, attempting conversion")
            macro_df.index = pd.to_datetime(macro_df.index)
            
        if not isinstance(price_df.index, pd.DatetimeIndex):
            self.logger.warning("Price data index is not DatetimeIndex, attempting conversion")
            price_df.index = pd.to_datetime(price_df.index)
            
        # Determine fill method from parameters
        fill_method = self.parameters.get('fill_method', 'ffill')
        
        # Reindex macro data to price data timeline
        aligned = macro_df.reindex(price_df.index, method=fill_method)
        
        # Handle any remaining NaNs based on parameter
        handle_missing = self.parameters.get('handle_missing', 'ffill')
        
        if handle_missing == 'ffill':
            aligned = aligned.ffill().bfill()
        elif handle_missing == 'nearest':
            aligned = aligned.interpolate(method='nearest')
        elif handle_missing == 'interpolate':
            aligned = aligned.interpolate(method='time')
        
        return aligned
    
    def _align_price_data(self, price_data: Dict[str, pd.DataFrame], 
                         context: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Align multiple price dataframes to common timeline
        
        Args:
            price_data: Dictionary of price dataframes by pair
            context: Pipeline context
            
        Returns:
            Dictionary of aligned price dataframes
        """
        # If only one pair, nothing to align
        if len(price_data) <= 1:
            return price_data
            
        self.logger.info(f"Aligning {len(price_data)} price dataframes")
        
        # Find common start and end dates
        common_start = None
        common_end = None
        
        for df in price_data.values():
            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.warning("Price data index is not DatetimeIndex, attempting conversion")
                df.index = pd.to_datetime(df.index)
                
            start = df.index.min()
            end = df.index.max()
            
            if common_start is None or start > common_start:
                common_start = start
                
            if common_end is None or end < common_end:
                common_end = end
        
        # Filter each dataframe to common date range
        result = {}
        for pair, df in price_data.items():
            result[pair] = df.loc[common_start:common_end]
            self.logger.info(f"Aligned {pair}: {len(result[pair])} rows")
            
        return result
    
    def _align_multi_dataframes(self, data_dict: Dict[str, pd.DataFrame], 
                               context: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Generic method to align multiple dataframes
        
        Args:
            data_dict: Dictionary of dataframes to align
            context: Pipeline context
            
        Returns:
            Dictionary of aligned dataframes
        """
        # This is a more generic version of _align_price_data
        if len(data_dict) <= 1:
            return data_dict
            
        # Determine common frequency, preferring the highest frequency
        freq_priority = ['S', 'T', 'min', '5min', '15min', '30min', 'H', '2H', '4H', 'D', 'W', 'M']
        target_freq = None
        
        # If align_to_highest_frequency parameter is True, find highest frequency
        if self.parameters.get('align_to_highest_frequency', True):
            for df in data_dict.values():
                if len(df) > 1 and isinstance(df.index, pd.DatetimeIndex):
                    # Infer frequency
                    inferred_freq = pd.infer_freq(df.index)
                    if inferred_freq:
                        # Check priority
                        if target_freq is None:
                            target_freq = inferred_freq
                        else:
                            idx1 = freq_priority.index(inferred_freq) if inferred_freq in freq_priority else len(freq_priority)
                            idx2 = freq_priority.index(target_freq) if target_freq in freq_priority else len(freq_priority)
                            if idx1 < idx2:  # Lower index means higher priority
                                target_freq = inferred_freq
        
        # If no frequency detected, just return original data
        if target_freq is None:
            self.logger.warning("Could not determine common frequency for alignment")
            return data_dict
            
        # Align to target frequency
        result = {}
        for key, df in data_dict.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                
            # Resample if needed
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq != target_freq:
                self.logger.info(f"Resampling {key} from {inferred_freq} to {target_freq}")
                result[key] = df.resample(target_freq).ffill()
            else:
                result[key] = df
                
        return result