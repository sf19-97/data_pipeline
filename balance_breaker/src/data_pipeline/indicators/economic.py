"""
Economic Indicators - Macroeconomic indicators and derived metrics

This component calculates economic indicators and derived metrics from macroeconomic data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.data_pipeline.base import BaseIndicator

@implements("IndicatorCalculator")
class EconomicIndicators(BaseIndicator):
    """
    Economic indicators calculator
    
    Parameters:
    -----------
    correlation_window : int
        Window for correlation calculations (default: 60)
    volatility_window : int
        Window for volatility calculations (default: 30)
    generate_spreads : bool
        Whether to generate yield spreads (default: True)
    generate_differentials : bool
        Whether to generate inflation differentials (default: True)
    policy_sensitivity : float
        Sensitivity factor for policy regime detection (default: 1.5)
    natural_rate_lookback : int
        Lookback period for natural rate estimation in days (default: 365)
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'correlation_window': 60,      # Window for correlation calculations
            'volatility_window': 30,       # Window for volatility calculations
            'generate_spreads': True,      # Generate yield spreads
            'generate_differentials': True, # Generate inflation differentials
            'policy_sensitivity': 1.5,     # Policy regime sensitivity
            'natural_rate_lookback': 365   # Lookback period for natural rate
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def calculate(self, data: Any, context: Dict[str, Any]) -> Any:
        """Calculate economic indicators
        
        Args:
            data: Input data (Dict containing 'price' and 'aligned_macro' if available,
                  or pd.DataFrame of macro data)
            context: Pipeline context
            
        Returns:
            Updated data with economic indicators
        """
        try:
            # Handle different input types
            if isinstance(data, dict) and 'aligned_macro' in data:
                # Process aligned macro data for each pair
                for pair, macro_df in data['aligned_macro'].items():
                    data['aligned_macro'][pair] = self._process_macro_data(macro_df, context, pair)
                return data
            
            elif isinstance(data, pd.DataFrame):
                # Single macro dataframe
                return self._process_macro_data(data, context)
                
            else:
                self.logger.warning(f"Unsupported data type for economic indicators: {type(data)}")
                return data
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={},
                subsystem='data_pipeline',
                component='EconomicIndicators'
            )
            raise
    
    def _process_macro_data(self, df: pd.DataFrame, context: Dict[str, Any], 
                           pair: Optional[str] = None) -> pd.DataFrame:
        """Process macro data to generate economic indicators
        
        Args:
            df: Macro dataframe
            context: Pipeline context
            pair: Currency pair (if applicable)
            
        Returns:
            DataFrame with additional economic indicators
        """
        # Skip processing if dataframe is empty
        if df.empty:
            return df
            
        self.logger.info(f"Calculating economic indicators{f' for {pair}' if pair else ''}")
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Determine base and quote currencies if pair is provided
        base_curr, quote_curr = None, None
        if pair:
            if len(pair) == 6:
                base_curr = pair[:3]
                quote_curr = pair[3:]
        
        # 1. Generate yield spreads (across countries)
        if self.parameters.get('generate_spreads', True):
            result = self._generate_yield_spreads(result, context, base_curr, quote_curr)
        
        # 2. Generate inflation differentials
        if self.parameters.get('generate_differentials', True):
            result = self._generate_inflation_differentials(result, context, base_curr, quote_curr)
        
        # 3. Calculate VIX correlations
        result = self._calculate_vix_correlations(result, context)
        
        # 4. Estimate monetary policy regimes if possible
        result = self._estimate_policy_regimes(result, context, base_curr, quote_curr)
        
        return result
    
    def _generate_yield_spreads(self, df: pd.DataFrame, context: Dict[str, Any],
                              base_curr: Optional[str] = None, 
                              quote_curr: Optional[str] = None) -> pd.DataFrame:
        """Generate yield spreads across different maturities
        
        Args:
            df: Macro dataframe
            context: Pipeline context
            base_curr: Base currency code (optional)
            quote_curr: Quote currency code (optional)
            
        Returns:
            DataFrame with yield spread indicators
        """
        # Look for existing 2Y and 10Y spreads
        spreads_2y = [col for col in df.columns if '2Y' in col]
        spreads_10y = [col for col in df.columns if '10Y' in col]
        
        if not spreads_2y and not spreads_10y:
            self.logger.debug("No yield spread columns found")
            return df
            
        # Calculate curve steepness (10Y-2Y spread) within countries
        for country_code in ['US', 'JP', 'AU', 'CA', 'EU', 'GB']:
            # Check if we have both 2Y and 10Y for this country
            country_2y = next((col for col in spreads_2y if f"{country_code}_2Y" in col), None)
            country_10y = next((col for col in spreads_10y if f"{country_code}_10Y" in col), None)
            
            if country_2y and country_10y:
                curve_name = f"{country_code}_CURVE_STEEPNESS"
                df[curve_name] = df[country_10y] - df[country_2y]
                self.logger.debug(f"Created {curve_name}")
        
        # Calculate cross-currency spreads if base and quote currencies are provided
        if base_curr and quote_curr:
            # Map currency codes to country codes
            curr_to_country = {
                'USD': 'US', 'JPY': 'JP', 'AUD': 'AU', 
                'CAD': 'CA', 'EUR': 'EU', 'GBP': 'GB'
            }
            
            base_country = curr_to_country.get(base_curr)
            quote_country = curr_to_country.get(quote_curr)
            
            if base_country and quote_country:
                # Try to find appropriate spreads
                for term in ['2Y', '10Y']:
                    base_col = next((col for col in df.columns if f"{base_country}_{term}" in col), None)
                    quote_col = next((col for col in df.columns if f"{quote_country}_{term}" in col), None)
                    
                    # If we have direct columns, calculate spread
                    if base_col and quote_col:
                        spread_name = f"{base_country}-{quote_country}_{term}"
                        df[spread_name] = df[base_col] - df[quote_col]
                        self.logger.debug(f"Created {spread_name}")
        
        return df
    
    def _generate_inflation_differentials(self, df: pd.DataFrame, context: Dict[str, Any],
                                       base_curr: Optional[str] = None, 
                                       quote_curr: Optional[str] = None) -> pd.DataFrame:
        """Generate inflation differentials across countries
        
        Args:
            df: Macro dataframe
            context: Pipeline context
            base_curr: Base currency code (optional)
            quote_curr: Quote currency code (optional)
            
        Returns:
            DataFrame with inflation differential indicators
        """
        # Look for inflation columns
        inflation_cols = [col for col in df.columns if 'CPI' in col or 'INFLATION' in col]
        
        if not inflation_cols:
            self.logger.debug("No inflation columns found")
            return df
            
        # Calculate real rate differentials (nominal rate - inflation)
        for country_code in ['US', 'JP', 'AU', 'CA', 'EU', 'GB']:
            # Check if we have both yields and inflation for this country
            country_yield = next((col for col in df.columns if f"{country_code}_10Y" in col), None)
            country_inflation = next((col for col in inflation_cols if country_code in col), None)
            
            if country_yield and country_inflation:
                real_rate_name = f"{country_code}_REAL_RATE"
                df[real_rate_name] = df[country_yield] - df[country_inflation]
                self.logger.debug(f"Created {real_rate_name}")
                
        # Calculate cross-currency inflation differentials if base and quote currencies are provided
        if base_curr and quote_curr:
            # Map currency codes to country codes
            curr_to_country = {
                'USD': 'US', 'JPY': 'JP', 'AUD': 'AU', 
                'CAD': 'CA', 'EUR': 'EU', 'GBP': 'GB'
            }
            
            base_country = curr_to_country.get(base_curr)
            quote_country = curr_to_country.get(quote_curr)
            
            if base_country and quote_country:
                # Try to find appropriate inflation indicators
                base_infl = next((col for col in inflation_cols if base_country in col), None)
                quote_infl = next((col for col in inflation_cols if quote_country in col), None)
                
                # If we have both, calculate differential
                if base_infl and quote_infl:
                    diff_name = f"{base_country}-{quote_country}_INFLATION_DIFF"
                    df[diff_name] = df[base_infl] - df[quote_infl]
                    self.logger.debug(f"Created {diff_name}")
                    
                # Also calculate real rate differential
                base_real = f"{base_country}_REAL_RATE"
                quote_real = f"{quote_country}_REAL_RATE"
                
                if base_real in df.columns and quote_real in df.columns:
                    real_diff_name = f"{base_country}-{quote_country}_REAL_RATE_DIFF"
                    df[real_diff_name] = df[base_real] - df[quote_real]
                    self.logger.debug(f"Created {real_diff_name}")
        
        return df
    
    def _calculate_vix_correlations(self, df: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Calculate rolling correlations between VIX and other indicators
        
        Args:
            df: Macro dataframe
            context: Pipeline context
            
        Returns:
            DataFrame with VIX correlation indicators
        """
        # Check if VIX is available
        if 'VIX' not in df.columns:
            self.logger.debug("VIX column not found, skipping correlation calculations")
            return df
            
        # Get correlation window size
        window = self.parameters.get('correlation_window', 60)
        
        # Calculate VIX changes
        df['VIX_CHANGE'] = df['VIX'].diff()
        
        # Calculate correlations with key indicators
        key_indicators = []
        
        # Find yield indicators
        yields = [col for col in df.columns if '10Y' in col or '2Y' in col]
        key_indicators.extend(yields)
        
        # Find inflation indicators
        inflation = [col for col in df.columns if 'CPI' in col or 'INFLATION' in col]
        key_indicators.extend(inflation)
        
        # Calculate correlations
        for indicator in key_indicators:
            if indicator in df.columns:
                # Calculate indicator changes
                indicator_change = df[indicator].diff()
                
                # Only proceed if we have enough non-NA values
                if (df['VIX_CHANGE'].notna() & indicator_change.notna()).sum() > window/2:
                    # Calculate rolling correlation
                    corr_name = f"VIX_{indicator}_CORR"
                    
                    # Use Spearman rank correlation for robustness
                    df[corr_name] = df['VIX_CHANGE'].rolling(window=window, min_periods=int(window/2)).corr(
                        indicator_change, method='spearman'
                    )
                    
                    self.logger.debug(f"Created {corr_name}")
        
        return df
    
    def _estimate_policy_regimes(self, df: pd.DataFrame, context: Dict[str, Any],
                              base_curr: Optional[str] = None, 
                              quote_curr: Optional[str] = None) -> pd.DataFrame:
        """Estimate monetary policy regimes
        
        Args:
            df: Macro dataframe
            context: Pipeline context
            base_curr: Base currency code (optional)
            quote_curr: Quote currency code (optional)
            
        Returns:
            DataFrame with policy regime indicators
        """
        # Map currency codes to country codes if provided
        target_countries = []
        
        if base_curr and quote_curr:
            curr_to_country = {
                'USD': 'US', 'JPY': 'JP', 'AUD': 'AU', 
                'CAD': 'CA', 'EUR': 'EU', 'GBP': 'GB'
            }
            
            if base_curr in curr_to_country:
                target_countries.append(curr_to_country[base_curr])
            if quote_curr in curr_to_country:
                target_countries.append(curr_to_country[quote_curr])
        
        # If no specific countries targeted, try for any available
        if not target_countries:
            target_countries = ['US', 'JP', 'AU', 'CA', 'EU', 'GB']
        
        # Get policy sensitivity parameter
        policy_sensitivity = self.parameters.get('policy_sensitivity', 1.5)
        natural_rate_lookback = self.parameters.get('natural_rate_lookback', 365)
        
        # Process each country
        for country in target_countries:
            # Look for indicators needed for regime estimation
            policy_rate = next((col for col in df.columns if f"{country}" in col and any(term in col for term in ['POLICY', 'FED', 'RATE'])), None)
            inflation = next((col for col in df.columns if f"{country}" in col and any(term in col for term in ['CPI', 'INFLATION'])), None)
            
            # We need at least inflation to proceed
            if inflation:
                # Estimate natural rate
                natural_rate_name = f"{country}_NATURAL_RATE"
                
                # If we already have the natural rate, use it
                if natural_rate_name in df.columns:
                    natural_rate = df[natural_rate_name]
                else:
                    # Simple estimate: average real rate over time
                    # Better methods exist but require more complex calculations
                    if policy_rate:
                        real_rate = df[policy_rate] - df[inflation]
                    else:
                        # Try to find a yield column
                        yield_col = next((col for col in df.columns if f"{country}" in col and '10Y' in col), None)
                        if yield_col:
                            real_rate = df[yield_col] - df[inflation]
                        else:
                            # Can't calculate real rate
                            continue
                    
                    # Smooth with long-term moving average
                    natural_rate = real_rate.rolling(window=natural_rate_lookback, min_periods=min(60, natural_rate_lookback//4)).mean()
                    df[natural_rate_name] = natural_rate
                
                # Calculate rate gap
                if policy_rate:
                    rate_gap_name = f"{country}_RATE_GAP"
                    df[rate_gap_name] = df[policy_rate] - natural_rate
                    
                    # Calculate probability of being at lower bound
                    lb_prob_name = f"{country}_LB_PROBABILITY"
                    # Use logistic function: 1/(1+exp(policy_sensitivity*gap))
                    df[lb_prob_name] = 1 / (1 + np.exp(policy_sensitivity * df[rate_gap_name]))
                    
                    # Binary regime indicator (0 = lower bound, 1 = normal)
                    # Following the typical threshold of (psi-1)/psi where psi is policy_sensitivity
                    threshold = (policy_sensitivity - 1) / policy_sensitivity
                    regime_name = f"{country}_REGIME"
                    df[regime_name] = (df[lb_prob_name] < threshold).astype(int)
                    
                    self.logger.debug(f"Created regime indicators for {country}")
        
        return df