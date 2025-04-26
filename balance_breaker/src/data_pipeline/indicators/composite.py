"""
Composite Indicators - Balance Breaker-specific composite indicators

This component calculates composite indicators specific to the Balance Breaker strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.data_pipeline.base import BaseIndicator

@implements("IndicatorCalculator")
class CompositeIndicators(BaseIndicator):
    """
    Calculates composite indicators specific to Balance Breaker strategy
    
    Parameters:
    -----------
    precession_window : int
        Window for precession calculation (default: 20)
    instability_window : int
        Window for instability calculation (default: 20)
    market_mood_window : int
        Window for market mood calculation (default: 20)
    natural_rate_lookback : int
        Lookback period for natural rate estimation (default: 365)
    natural_rate_smoothing : int
        Smoothing period for natural rate (default: 60)
    policy_psi : float
        Policy rule parameter (default: 1.5)
    corr_threshold : float
        VIX-inflation correlation threshold (default: -0.2)
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'precession_window': 20,       # Window for precession calculation
            'instability_window': 20,      # Window for instability calculation
            'market_mood_window': 20,      # Window for market mood calculation
            'natural_rate_lookback': 365,  # Lookback period for natural rate estimation
            'natural_rate_smoothing': 60,  # Smoothing period for natural rate
            'policy_psi': 1.5,             # Policy rule parameter
            'corr_threshold': -0.2         # VIX-inflation correlation threshold
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def calculate(self, data: Any, context: Dict[str, Any]) -> Any:
        """Calculate composite indicators
        
        Args:
            data: Input data (Dict with 'price' and 'aligned_macro')
            context: Pipeline context
            
        Returns:
            Updated data with composite indicators
        """
        try:
            # Ensure we have the right data structure
            if not isinstance(data, dict) or 'aligned_macro' not in data or 'price' not in data:
                self.logger.warning("Invalid data structure for composite indicators")
                return data
                
            # Process each pair
            for pair in data['price'].keys():
                if pair in data['aligned_macro']:
                    try:
                        # Calculate Balance Breaker specific indicators
                        bb_indicators = self._calculate_balance_breaker_indicators(
                            price_df=data['price'][pair],
                            macro_df=data['aligned_macro'][pair],
                            pair=pair,
                            context=context
                        )
                        
                        # Add indicators to data
                        data['aligned_macro'][pair] = pd.concat([
                            data['aligned_macro'][pair], 
                            bb_indicators
                        ], axis=1)
                        
                        self.logger.info(f"Added composite indicators for {pair}")
                        
                    except Exception as e:
                        self.error_handler.handle_error(
                            e,
                            context={'pair': pair},
                            subsystem='data_pipeline',
                            component='CompositeIndicators'
                        )
            
            return data
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={},
                subsystem='data_pipeline',
                component='CompositeIndicators'
            )
            raise
    
    def _calculate_balance_breaker_indicators(self, price_df: pd.DataFrame, 
                                            macro_df: pd.DataFrame,
                                            pair: str,
                                            context: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Balance Breaker specific indicators
        
        Args:
            price_df: Price dataframe
            macro_df: Macro dataframe
            pair: Currency pair
            context: Pipeline context
            
        Returns:
            DataFrame with composite indicators
        """
        # Create an empty DataFrame with the same index
        indicators = pd.DataFrame(index=price_df.index)
        
        # Get base and quote currencies
        base_curr, quote_curr = pair[:3], pair[3:] if len(pair) == 6 else (None, None)
        
        # Map currency codes to country codes
        curr_to_country = {
            'USD': 'US', 'JPY': 'JP', 'AUD': 'AU', 
            'CAD': 'CA', 'EUR': 'EU', 'GBP': 'GB'
        }
        
        base_country = curr_to_country.get(base_curr, '')
        quote_country = curr_to_country.get(quote_curr, '')
        
        # 1. Calculate precession
        indicators['precession'] = self._calculate_precession(macro_df)
        
        # 2. Calculate market mood
        indicators['market_mood'] = self._calculate_market_mood(macro_df, price_df, base_country, quote_country)
        
        # 3. Calculate instability
        indicators['instability'] = self._calculate_instability(macro_df, price_df)
        
        # 4. Get regime information
        indicators = self._add_regime_info(indicators, macro_df, base_country, quote_country)
        
        # 5. Calculate correlation impacts
        indicators = self._add_correlation_impacts(indicators, macro_df, base_country, quote_country)
        
        return indicators
    
    def _calculate_precession(self, macro_df: pd.DataFrame) -> pd.Series:
        """Calculate precession indicator
        
        Args:
            macro_df: Macro dataframe
            
        Returns:
            Series with precession values
        """
        window = self.parameters.get('precession_window', 20)
        
        # Find yield columns for precession calculation
        yield_cols = [col for col in macro_df.columns if any(term in col for term in ['2Y', '10Y'])]
        
        if not yield_cols:
            return pd.Series(0, index=macro_df.index)
        
        # Calculate rate of change in yields
        roc_series = []
        for col in yield_cols:
            # ROC over window period
            roc = macro_df[col].diff(window) / window
            roc_series.append(roc)
        
        # Combine ROC series (mean, weighted by correlation with each other)
        if len(roc_series) == 1:
            combined_roc = roc_series[0]
        else:
            # Calculate weights based on correlation with other series
            weights = []
            for i, series in enumerate(roc_series):
                # Get average absolute correlation with other series
                corrs = [series.corr(other) for j, other in enumerate(roc_series) if i != j]
                if corrs:
                    weight = np.mean(np.abs(corrs))
                    weights.append(weight)
                else:
                    weights.append(1.0)
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
            # Create combined weighted ROC
            combined_roc = pd.Series(0, index=macro_df.index)
            for i, series in enumerate(roc_series):
                combined_roc += series * weights[i]
        
        # Apply tanh to scale and normalize
        precession = np.tanh(combined_roc * 5)
        
        return precession
    
    def _calculate_market_mood(self, macro_df: pd.DataFrame, price_df: pd.DataFrame,
                            base_country: str, quote_country: str) -> pd.Series:
        """Calculate market mood indicator
        
        Args:
            macro_df: Macro dataframe
            price_df: Price dataframe
            base_country: Base currency country code
            quote_country: Quote currency country code
            
        Returns:
            Series with market mood values
        """
        window = self.parameters.get('market_mood_window', 20)
        
        # Find relevant columns for mood calculation
        mood_components = []
        
        # 1. Yield spreads (if available)
        if base_country and quote_country:
            spread_col = f"{base_country}-{quote_country}_10Y"
            if spread_col in macro_df.columns:
                # Normalize with tanh and recent change
                spread_change = macro_df[spread_col].diff(window)
                mood_components.append(np.tanh(spread_change))
        
        # 2. Inflation differentials (if available)
        if base_country and quote_country:
            infl_col = f"{base_country}-{quote_country}_INFLATION_DIFF"
            if infl_col in macro_df.columns:
                # Normalize with tanh
                infl_diff = macro_df[infl_col]
                mood_components.append(np.tanh(infl_diff / 2))
        
        # 3. Price momentum
        price_roc = price_df['close'].pct_change(window)
        mood_components.append(np.tanh(price_roc * 10))  # Scale for sensitivity
        
        # Combine mood components
        if not mood_components:
            return pd.Series(0, index=macro_df.index)
            
        # Weight components (more sophisticated weighting could be done here)
        weights = [0.4, 0.3, 0.3][:len(mood_components)]
        weights = [w / sum(weights) for w in weights]
        
        # Calculate weighted mood
        mood = pd.Series(0, index=macro_df.index)
        for i, component in enumerate(mood_components):
            # Ensure component has the same index
            reindexed = component.reindex(mood.index)
            mood += reindexed * weights[i]
        
        return mood
    
    def _calculate_instability(self, macro_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.Series:
        """Calculate instability indicator
        
        Args:
            macro_df: Macro dataframe
            price_df: Price dataframe
            
        Returns:
            Series with instability values
        """
        window = self.parameters.get('instability_window', 20)
        
        # Components of instability
        instability_components = []
        
        # 1. VIX (if available)
        if 'VIX' in macro_df.columns:
            # VIX normalized by its own recent range
            vix = macro_df['VIX']
            vix_min = vix.rolling(window=window).min()
            vix_max = vix.rolling(window=window).max()
            vix_range = vix_max - vix_min
            
            # Normalize to 0-1 range within window
            vix_norm = (vix - vix_min) / vix_range.replace(0, 1)
            instability_components.append(vix_norm)
        
        # 2. Price volatility
        returns = price_df['close'].pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(window)
        
        # Normalize volatility relative to its own history
        vol_norm = volatility / volatility.rolling(window=window*5).mean().replace(0, volatility.mean())
        instability_components.append(vol_norm)
        
        # 3. Rate change volatility
        rate_cols = [col for col in macro_df.columns if '10Y' in col or '2Y' in col]
        if rate_cols:
            # Use the most volatile rate series
            rate_vols = []
            for col in rate_cols:
                rate_change = macro_df[col].diff()
                rate_vol = rate_change.rolling(window=window).std() * np.sqrt(window)
                rate_vols.append(rate_vol)
            
            # Combine rate volatilities
            rate_vol_combined = pd.concat(rate_vols, axis=1).mean(axis=1)
            rate_vol_norm = rate_vol_combined / rate_vol_combined.rolling(window=window*5).mean().replace(0, rate_vol_combined.mean())
            instability_components.append(rate_vol_norm)
        
        # Combine components
        if not instability_components:
            return pd.Series(1, index=macro_df.index)  # Default to neutral
            
        # Calculate combined instability
        instability = pd.concat(instability_components, axis=1).mean(axis=1)
        
        # Rescale for typical range of 0.5-3.0
        instability = instability * 2.5 + 0.5
        
        return instability
    
    def _add_regime_info(self, indicators: pd.DataFrame, macro_df: pd.DataFrame,
                       base_country: str, quote_country: str) -> pd.DataFrame:
        """Add regime information to indicators
        
        Args:
            indicators: Indicators dataframe
            macro_df: Macro dataframe
            base_country: Base currency country code
            quote_country: Quote currency country code
            
        Returns:
            Updated indicators dataframe
        """
        # Look for existing regime indicators
        base_regime = f"{base_country}_REGIME" if base_country else None
        quote_regime = f"{quote_country}_REGIME" if quote_country else None
        
        if base_regime and base_regime in macro_df.columns:
            # Copy regime directly
            indicators['regime'] = macro_df[base_regime].map({
                1: "TARGET_EQUILIBRIUM",
                0: "LOWER_BOUND_RISK"
            }).fillna("TARGET_EQUILIBRIUM")
            
            # Copy lower bound probability if available
            lb_prob_col = f"{base_country}_LB_PROBABILITY"
            if lb_prob_col in macro_df.columns:
                indicators['lower_bound_probability'] = macro_df[lb_prob_col]
                
        elif quote_regime and quote_regime in macro_df.columns:
            # Use quote currency regime as fallback
            indicators['regime'] = macro_df[quote_regime].map({
                1: "TARGET_EQUILIBRIUM",
                0: "LOWER_BOUND_RISK"
            }).fillna("TARGET_EQUILIBRIUM")
            
            # Copy lower bound probability if available
            lb_prob_col = f"{quote_country}_LB_PROBABILITY"
            if lb_prob_col in macro_df.columns:
                indicators['lower_bound_probability'] = macro_df[lb_prob_col]
                
        else:
            # Default to target equilibrium if no regime info available
            indicators['regime'] = "TARGET_EQUILIBRIUM"
            indicators['lower_bound_probability'] = 0.0
        
        return indicators
    
    def _add_correlation_impacts(self, indicators: pd.DataFrame, macro_df: pd.DataFrame,
                              base_country: str, quote_country: str) -> pd.DataFrame:
        """Add correlation impact indicators
        
        Args:
            indicators: Indicators dataframe
            macro_df: Macro dataframe
            base_country: Base currency country code
            quote_country: Quote currency country code
            
        Returns:
            Updated indicators dataframe
        """
        # Get correlation threshold parameter
        corr_threshold = self.parameters.get('corr_threshold', -0.2)
        
        # Look for VIX correlation columns
        vix_cols = [col for col in macro_df.columns if 'VIX_' in col and 'CORR' in col]
        
        # Add specific correlations if available
        if base_country:
            # VIX-Inflation correlation
            vix_infl_col = next((col for col in vix_cols if f"{base_country}" in col and 'INFLATION' in col), None)
            if vix_infl_col:
                indicators['vix_inflation_correlation'] = macro_df[vix_infl_col]
            
            # VIX-Rates correlation
            vix_rates_col = next((col for col in vix_cols if f"{base_country}" in col and ('10Y' in col or '2Y' in col)), None)
            if vix_rates_col:
                indicators['vix_rates_correlation'] = macro_df[vix_rates_col]
        
        # If we don't have specific correlations, try using any available
        if 'vix_inflation_correlation' not in indicators.columns:
            # Use first available VIX-inflation correlation
            vix_infl_col = next((col for col in vix_cols if 'INFLATION' in col or 'CPI' in col), None)
            if vix_infl_col:
                indicators['vix_inflation_correlation'] = macro_df[vix_infl_col]
            else:
                indicators['vix_inflation_correlation'] = 0.0
                
        if 'vix_rates_correlation' not in indicators.columns:
            # Use first available VIX-rates correlation
            vix_rates_col = next((col for col in vix_cols if '10Y' in col or '2Y' in col), None)
            if vix_rates_col:
                indicators['vix_rates_correlation'] = macro_df[vix_rates_col]
            else:
                indicators['vix_rates_correlation'] = 0.0
        
        # Add indicator for significant VIX-inflation correlation
        indicators['strong_vix_inflation_corr'] = (indicators['vix_inflation_correlation'] < corr_threshold).astype(int)
        
        return indicators