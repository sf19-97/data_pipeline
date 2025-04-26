"""
Feature Creator - Data feature engineering processor

This component generates trading features from price and macro data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.data_pipeline.base import BaseProcessor

@implements("DataProcessor")
class FeatureCreator(BaseProcessor):
    """
    Creator for trading features from price and macro data
    
    Parameters:
    -----------
    return_periods : List[int]
        Periods for calculating returns (default: [1, 5, 10, 20])
    ma_periods : List[int]
        Periods for moving averages (default: [5, 10, 20, 50])
    create_lag_features : bool
        Whether to create lagged features (default: True)
    lag_periods : List[int]
        Periods for lagged features (default: [1, 2, 3])
    rolling_windows : List[int]
        Window sizes for rolling statistics (default: [5, 10, 20])
    normalize_features : bool
        Whether to normalize features (default: False)
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'return_periods': [1, 5, 10, 20],
            'ma_periods': [5, 10, 20, 50],
            'create_lag_features': True,
            'lag_periods': [1, 2, 3],
            'rolling_windows': [5, 10, 20],
            'normalize_features': False,
            'include_technical': True
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def process_data(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process data by creating features
        
        Args:
            data: Input data (Dict[str, pd.DataFrame] for price, 
                  or Dict with 'price' and 'aligned_macro')
            context: Pipeline context
            
        Returns:
            Data with additional features
        """
        try:
            data_type = context.get('data_type', 'price')
            
            # Handle different input types
            if isinstance(data, dict) and 'price' in data and 'aligned_macro' in data:
                # Process both price and macro data
                for pair in data['price'].keys():
                    if pair in data['aligned_macro']:
                        # Create price features
                        data['price'][pair] = self._create_price_features(
                            data['price'][pair], context, pair
                        )
                        
                        # Create combined features
                        data['aligned_macro'][pair] = self._create_macro_features(
                            data['aligned_macro'][pair], 
                            price_df=data['price'][pair],
                            context=context, 
                            pair=pair
                        )
                
                return data
                
            elif isinstance(data, dict) and all(isinstance(df, pd.DataFrame) for df in data.values()):
                # Dictionary of price dataframes by pair
                result = {}
                for pair, df in data.items():
                    result[pair] = self._create_price_features(df, context, pair)
                return result
                
            elif isinstance(data, pd.DataFrame):
                # Single dataframe (price or macro)
                if data_type == 'price':
                    return self._create_price_features(data, context)
                elif data_type == 'macro':
                    return self._create_macro_features(data, None, context)
                
            self.logger.warning(f"Unsupported data type for feature creation: {type(data)}")
            return data
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'data_type': context.get('data_type', 'unknown')},
                subsystem='data_pipeline',
                component='FeatureCreator'
            )
            raise
    
    def _create_price_features(self, df: pd.DataFrame, context: Dict[str, Any],
                             pair: Optional[str] = None) -> pd.DataFrame:
        """Create features from price data
        
        Args:
            df: Price dataframe
            context: Pipeline context
            pair: Currency pair name (optional)
            
        Returns:
            DataFrame with additional features
        """
        if df.empty:
            return df
            
        # Work with a copy
        result = df.copy()
        
        # 1. Calculate returns for different periods
        for period in self.parameters.get('return_periods', [1, 5, 10, 20]):
            # Simple returns
            result[f'return_{period}'] = df['close'].pct_change(period)
            
            # Log returns (better statistical properties)
            result[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        # 2. Create moving averages
        if self.parameters.get('include_technical', True):
            for period in self.parameters.get('ma_periods', [5, 10, 20, 50]):
                result[f'ma_{period}'] = df['close'].rolling(window=period).mean()
                
                # Moving average crossovers
                if period > 5:
                    result[f'ma_5_cross_{period}'] = (
                        result['ma_5'] > result[f'ma_{period}']
                    ).astype(int)
        
        # 3. Create rolling statistics
        for window in self.parameters.get('rolling_windows', [5, 10, 20]):
            # Volatility (standard deviation of returns)
            result[f'volatility_{window}'] = result['return_1'].rolling(window=window).std()
            
            # Momentum (sum of returns)
            result[f'momentum_{window}'] = result['return_1'].rolling(window=window).sum()
            
            # Min/max normalization over window
            result[f'normed_price_{window}'] = (
                (df['close'] - df['close'].rolling(window=window).min()) /
                (df['close'].rolling(window=window).max() - df['close'].rolling(window=window).min())
            )
        
        # 4. Create lagged features if enabled
        if self.parameters.get('create_lag_features', True):
            for period in self.parameters.get('lag_periods', [1, 2, 3]):
                # Lagged price and returns
                result[f'close_lag_{period}'] = df['close'].shift(period)
                result[f'return_lag_{period}'] = result['return_1'].shift(period)
                
                # Return sign
                result[f'return_sign_lag_{period}'] = np.sign(result['return_1'].shift(period))
        
        # 5. Create volatility indicators
        if 'high' in df.columns and 'low' in df.columns:
            # High-Low range
            result['high_low_range'] = (df['high'] - df['low']) / df['close']
            
            # Average true range components
            result['tr1'] = df['high'] - df['low']  # Current high-low
            result['tr2'] = abs(df['high'] - df['close'].shift(1))  # Current high - prev close
            result['tr3'] = abs(df['low'] - df['close'].shift(1))  # Current low - prev close
            
            # True range and ATR
            result['true_range'] = result[['tr1', 'tr2', 'tr3']].max(axis=1)
            result['atr_14'] = result['true_range'].rolling(window=14).mean()
            
            # Clean up intermediate columns
            result = result.drop(['tr1', 'tr2', 'tr3'], axis=1)
        
        # 6. Create price patterns and candle features
        result['body_size'] = abs(df['close'] - df['open']) / df['close']
        result['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        result['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # 7. Normalize features if requested
        if self.parameters.get('normalize_features', False):
            # Get list of numeric columns to normalize
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            
            # Skip price columns, only normalize derived features
            skip_cols = ['open', 'high', 'low', 'close', 'volume']
            cols_to_normalize = [col for col in numeric_cols if col not in skip_cols]
            
            # Z-score normalization
            for col in cols_to_normalize:
                mean = result[col].mean()
                std = result[col].std()
                
                if std > 0:  # Avoid division by zero
                    result[col] = (result[col] - mean) / std
        
        # Log feature creation
        self.logger.info(f"Created {len(result.columns) - len(df.columns)} features for price data{f' ({pair})' if pair else ''}")
        
        return result
    
    def _create_macro_features(self, macro_df: pd.DataFrame, price_df: Optional[pd.DataFrame],
                             context: Dict[str, Any], pair: Optional[str] = None) -> pd.DataFrame:
        """Create features from macro data
        
        Args:
            macro_df: Macro dataframe
            price_df: Price dataframe (optional)
            context: Pipeline context
            pair: Currency pair name (optional)
            
        Returns:
            DataFrame with additional features
        """
        if macro_df.empty:
            return macro_df
            
        # Work with a copy
        result = macro_df.copy()
        
        # 1. Calculate changes for macro indicators
        for col in macro_df.columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(macro_df[col]):
                continue
                
            # Calculate changes
            result[f'{col}_change'] = macro_df[col].diff()
            result[f'{col}_pct_change'] = macro_df[col].pct_change()
            
            # Calculate rolling changes
            for window in self.parameters.get('rolling_windows', [5, 10, 20]):
                if len(macro_df) > window:  # Ensure enough data
                    result[f'{col}_slope_{window}'] = (
                        (macro_df[col] - macro_df[col].shift(window)) / window
                    )
        
        # 2. Calculate relative metrics if price data is available
        if price_df is not None and 'close' in price_df.columns:
            # Align price data to macro data index
            price_series = price_df['close'].reindex(macro_df.index)
            
            # Create price-macro ratios for important indicators
            for col in macro_df.columns:
                # Only create ratios for specific indicator types
                if any(indicator in col.lower() for indicator in 
                     ['rate', 'yield', 'inflation', 'cpi', 'gdp', 'unemployment']):
                    result[f'price_to_{col}'] = price_series / macro_df[col]
        
        # 3. Create interaction features for key pairs of variables
        # This finds correlations between macro indicators
        if len(macro_df.columns) > 1:
            key_indicators = []
            
            # Find key indicators
            for indicator_type in ['rate', 'inflation', 'vix', 'gdp', 'unemployment']:
                for col in macro_df.columns:
                    if indicator_type in col.lower() and col not in key_indicators:
                        key_indicators.append(col)
                        break  # Take the first one of each type
            
            # Create interactions between key indicators
            for i, col1 in enumerate(key_indicators):
                for col2 in key_indicators[i+1:]:
                    # Only proceed if both columns are numeric
                    if (pd.api.types.is_numeric_dtype(macro_df[col1]) and 
                        pd.api.types.is_numeric_dtype(macro_df[col2])):
                        # Ratio between indicators
                        ratio_name = f'{col1}_to_{col2}'
                        if ratio_name not in result.columns:
                            result[ratio_name] = macro_df[col1] / macro_df[col2]
        
        # 4. Calculate lagged features for macro data
        if self.parameters.get('create_lag_features', True):
            for period in self.parameters.get('lag_periods', [1, 2, 3]):
                for col in key_indicators if 'key_indicators' in locals() else []:
                    result[f'{col}_lag_{period}'] = macro_df[col].shift(period)
        
        # Log feature creation
        self.logger.info(f"Created {len(result.columns) - len(macro_df.columns)} features for macro data{f' ({pair})' if pair else ''}")
        
        return result