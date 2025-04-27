"""
Confidence Filter

This component filters signals based on confidence and strength levels.
"""

from typing import Dict, List, Any, Optional

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.signal.base import BaseSignalFilter
from balance_breaker.src.signal.signal_models import Signal

@implements("SignalFilter")
class ConfidenceFilter(BaseSignalFilter):
    """
    Filters signals based on confidence level
    
    Parameters:
    -----------
    min_confidence : int
        Minimum confidence level (default: 2)
    min_strength : int
        Minimum strength level (default: 2)
    require_both : bool
        Whether to require both confidence and strength thresholds (default: False)
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'min_confidence': 2,
            'min_strength': 2,
            'require_both': False
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
            min_confidence = self.parameters.get('min_confidence', 2)
            min_strength = self.parameters.get('min_strength', 2)
            require_both = self.parameters.get('require_both', False)
            
            # Override from context if provided
            if 'min_confidence' in context:
                min_confidence = context['min_confidence']
            if 'min_strength' in context:
                min_strength = context['min_strength']
            if 'require_both' in context:
                require_both = context['require_both']
            
            # Get signal confidence and strength
            confidence = signal.metadata.confidence.value
            strength = signal.strength.value
            
            # Apply filters
            confidence_ok = confidence >= min_confidence
            strength_ok = strength >= min_strength
            
            if require_both:
                return confidence_ok and strength_ok
            else:
                return confidence_ok or strength_ok
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'signal_id': signal.id},
                subsystem='signal',
                component='ConfidenceFilter'
            )
            return False