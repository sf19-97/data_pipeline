"""
Timeframe Filter

This component filters signals based on their timeframe.
"""

from typing import Dict, List, Any, Optional

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.signal.base import BaseSignalFilter
from balance_breaker.src.signal.signal_models import Signal, Timeframe

@implements("SignalFilter")
class TimeframeFilter(BaseSignalFilter):
    """
    Filters signals based on timeframe
    
    Parameters:
    -----------
    allowed_timeframes : List[str]
        List of allowed timeframes (default: all timeframes)
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'allowed_timeframes': [tf.value for tf in Timeframe]
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def passes_filter(self, signal: Signal, context: Dict[str, Any]) -> bool:
        """Check if a signal passes the filter
        
        Args:
            signal: Signal to check
            context: Filtering context
            
        Returns:
            True if signal passes filter, False otherwise
        """
        try:
            # Get parameters
            allowed_timeframes = self.parameters.get('allowed_timeframes', [tf.value for tf in Timeframe])
            
            # Override from context if provided
            if 'allowed_timeframes' in context:
                allowed_timeframes = context['allowed_timeframes']
            
            # Check if signal timeframe is in allowed timeframes
            signal_timeframe = signal.timeframe.value if hasattr(signal.timeframe, 'value') else signal.timeframe
            return signal_timeframe in allowed_timeframes
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'signal_id': signal.id},
                subsystem='signal',
                component='TimeframeFilter'
            )
            return False