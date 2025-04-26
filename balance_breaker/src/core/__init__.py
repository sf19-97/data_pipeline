# Export core modules
from .error_handling import ErrorHandler, BalanceBreakerError, ErrorSeverity, ErrorCategory
from .parameter_manager import ParameterizedComponent, ParameterManager
from .interface_registry import interface, implements, registry
from .integration_tools import event_bus, service_registry

__all__ = [
    'ErrorHandler',
    'BalanceBreakerError',
    'ErrorSeverity',
    'ErrorCategory',
    'ParameterizedComponent',
    'ParameterManager',
    'interface',
    'implements',
    'registry',
    'event_bus',
    'service_registry'
]