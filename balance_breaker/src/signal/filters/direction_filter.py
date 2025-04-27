"""
Direction Filter

This component filters signals based on their direction.
"""

from typing import Dict, List, Any, Optional

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.signal.base import BaseSignalFilter
from balance_breaker.src.signal.signal_models import Signal, SignalDirection

@implements("SignalFilter")
class DirectionFilter(BaseSignalFilter):
    """
    Filters signals based on direction
    
    Parameters:
    -----------
    allowed_directions : List[str]
        List of allowed directions (default: all directions)
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'allowed_directions': [direction.value for direction in SignalDirection]
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
            allowed_directions = self.parameters.get('allowed_directions', [direction.value for direction in SignalDirection])
            
            # Override from context if provided
            if 'allowed_directions' in context:
                allowed_directions = context['allowed_directions']
            
            # Check if signal direction is in allowed directions
            signal_direction = signal.direction.value if hasattr(signal.direction, 'value') else signal.direction
            return signal_direction in allowed_directions
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'signal_id': signal.id},
                subsystem='signal',
                component='DirectionFilter'
            )
            return False