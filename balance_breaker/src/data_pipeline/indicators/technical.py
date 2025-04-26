"""
Technical Indicators - Price-based technical analysis indicators

This component calculates a variety of technical analysis indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List, Optional, Tuple

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.data_pipeline.base import BaseIndicator

@implements("IndicatorCalculator")
class TechnicalIndicators(BaseIndicator):
    """
    Technical analysis indicators calculator
    
    Parameters:
    -----------
    sma_periods : List[int]
        Periods for Simple Moving Average (default: [10, 20, 50, 200])
    ema_periods : List[int]
        Periods for Exponential Moving Average (default: [5, 12, 26])
    rsi_period : int
        Period for Relative Strength Index (default: 14)
    macd_params : Tuple[int, int, int]
        MACD parameters (fast, slow, signal) (default: (12, 26, 9))
    bbands_params : Tuple[int, int]
        Bollinger Bands parameters (period, std_dev) (default: (20, 2))
    atr_period : int
        Average True Range period (default: 14)
    stoch_params : Tuple[int, int, int]
        Stochastic parameters (k_period, k_slowing, d_period) (default: (14, 3, 3))
    generate_all : bool
        Whether to generate all indicators (default: False)
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'sma_periods': [10, 20, 50, 200],
            'ema_periods': [5, 12, 26],
            'rsi_period': 14,
            'macd_params': (12, 26, 9),
            'bbands_params': (20, 2),
            'atr_period': 14,
            'stoch_params': (14, 3, 3),
            'generate_all': False
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def calculate(self, data: Any, context: Dict[str, Any]) -> Any:
        """Calculate technical indicators
        
        Args:
            data: Input data (Dict containing 'price' data by pair,
                  or Dict[str, pd.DataFrame] of price data by pair,
                  or pd.DataFrame of price data)
            context: Pipeline context
            
        Returns:
            Updated data with technical indicators
        """
        try:
            # Handle different input types
            if isinstance(data, dict) and 'price' in data:
                # Process price data for each pair
                for pair, price_df in data['price'].items():
                    data['price'][pair] = self._process_price_data(price_df, context, pair)
                return data
                
            elif isinstance(data, dict) and all(isinstance(df, pd.DataFrame) for df in data.values()):
                # Dictionary of price dataframes by pair
                for pair, price_df in data.items():
                    data[pair] = self._process_price_data(price_df, context, pair)
                return data
                
            elif isinstance(data, pd.DataFrame):
                # Single price dataframe
                return self._process_price_data(data, context)
                
            else:
                self.logger.warning(f"Unsupported data type for technical indicators: {type(data)}")
                return data
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={},
                subsystem='data_pipeline',
                component='TechnicalIndicators'
            )
            raise
    
    def _process_price_data(self, df: pd.DataFrame, context: Dict[str, Any],
                          pair: Optional[str] = None) -> pd.DataFrame:
        """Process price data to calculate technical indicators
        
        Args:
            df: Price dataframe with OHLC columns
            context: Pipeline context
            pair: Currency pair (if applicable)
            
        Returns:
            DataFrame with technical indicators
        """
        # Verify we have OHLC data
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            self.logger.warning(f"Missing required OHLC columns in price data{f' for {pair}' if pair else ''}")
            return df
            
        # Skip processing if dataframe is empty
        if df.empty:
            return df
            
        self.logger.info(f"Calculating technical indicators{f' for {pair}' if pair else ''}")
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Get selected indicators from context or use parameters
        indicators = context.get('technical_indicators', [])
        generate_all = self.parameters.get('generate_all', False)
        
        # If specific indicators are requested, only calculate those
        if indicators and not generate_all:
            for indicator in indicators:
                result = self._calculate_indicator(result, indicator, context)
        else:
            # Calculate all default indicators
            result = self._calculate_moving_averages(result)
            result = self._calculate_rsi(result)
            result = self._calculate_macd(result)
            result = self._calculate_bollinger_bands(result)
            result = self._calculate_atr(result)
            result = self._calculate_stochastic(result)
            
        return result
    
    def _calculate_indicator(self, df: pd.DataFrame, indicator: str, 
                           context: Dict[str, Any]) -> pd.DataFrame:
        """Calculate a specific indicator
        
        Args:
            df: Price dataframe
            indicator: Indicator name
            context: Pipeline context
            
        Returns:
            DataFrame with the indicator added
        """
        indicator_type = indicator.lower()
        
        if 'sma' in indicator_type or 'ema' in indicator_type:
            return self._calculate_moving_averages(df)
        elif 'rsi' in indicator_type:
            return self._calculate_rsi(df)
        elif 'macd' in indicator_type:
            return self._calculate_macd(df)
        elif 'boll' in indicator_type:
            return self._calculate_bollinger_bands(df)
        elif 'atr' in indicator_type:
            return self._calculate_atr(df)
        elif 'stoch' in indicator_type:
            return self._calculate_stochastic(df)
        else:
            self.logger.warning(f"Unknown indicator type: {indicator}")
            return df
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Simple and Exponential Moving Averages
        
        Args:
            df: Price dataframe
            
        Returns:
            DataFrame with SMA and EMA columns added
        """
        price = df['close']
        
        # Calculate SMAs
        for period in self.parameters.get('sma_periods', [10, 20, 50, 200]):
            df[f'SMA_{period}'] = price.rolling(window=period).mean()
            
        # Calculate EMAs
        for period in self.parameters.get('ema_periods', [5, 12, 26]):
            df[f'EMA_{period}'] = price.ewm(span=period, adjust=False).mean()
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Relative Strength Index (RSI)
        
        Args:
            df: Price dataframe
            
        Returns:
            DataFrame with RSI column added
        """
        period = self.parameters.get('rsi_period', 14)
        price = df['close']
        
        # Calculate price changes
        delta = price.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate relative strength (avoid divide by zero)
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        
        # Calculate RSI
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Moving Average Convergence Divergence (MACD)
        
        Args:
            df: Price dataframe
            
        Returns:
            DataFrame with MACD columns added
        """
        fast_period, slow_period, signal_period = self.parameters.get('macd_params', (12, 26, 9))
        price = df['close']
        
        # Calculate fast and slow EMAs
        fast_ema = price.ewm(span=fast_period, adjust=False).mean()
        slow_ema = price.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        df['MACD'] = fast_ema - slow_ema
        
        # Calculate signal line
        df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands
        
        Args:
            df: Price dataframe
            
        Returns:
            DataFrame with Bollinger Bands columns added
        """
        period, std_dev = self.parameters.get('bbands_params', (20, 2))
        price = df['close']
        
        # Calculate middle band (SMA)
        df['BB_Middle'] = price.rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = price.rolling(window=period).std()
        
        # Calculate upper and lower bands
        df['BB_Upper'] = df['BB_Middle'] + (rolling_std * std_dev)
        df['BB_Lower'] = df['BB_Middle'] - (rolling_std * std_dev)
        
        # Calculate bandwidth and %B
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_B'] = (price - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'].replace(0, np.finfo(float).eps))
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range (ATR)
        
        Args:
            df: Price dataframe
            
        Returns:
            DataFrame with ATR column added
        """
        period = self.parameters.get('atr_period', 14)
        
        # Calculate true range
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        df['ATR'] = df['TR'].rolling(window=period).mean()
        
        return df
    
    def _calculate_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator
        
        Args:
            df: Price dataframe
            
        Returns:
            DataFrame with Stochastic Oscillator columns added
        """
        k_period, k_slowing, d_period = self.parameters.get('stoch_params', (14, 3, 3))
        
        # Calculate %K
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        # Avoid division by zero
        denominator = high_max - low_min
        denominator = denominator.replace(0, np.finfo(float).eps)
        
        # Raw %K
        df['Stoch_K_Raw'] = 100 * (df['close'] - low_min) / denominator
        
        # Apply slowing to get %K
        df['Stoch_K'] = df['Stoch_K_Raw'].rolling(window=k_slowing).mean()
        
        # Calculate %D
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
        
        return df