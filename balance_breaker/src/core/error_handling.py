"""
Error Handling System

This module provides a standardized error handling system for Balance Breaker.
It ensures consistent error reporting, logging, and handling across all subsystems.
"""

import logging
import traceback
import sys
from typing import Dict, Any, Optional, List, Type, Union
from enum import Enum
import datetime
from contextlib import contextmanager 


class ErrorSeverity(Enum):
    """Error severity levels"""
    DEBUG = 10      # Informational, debug-level errors
    INFO = 20       # Informational, but noteworthy
    WARNING = 30    # Warning, potential issues but not critical
    ERROR = 40      # Error, operation failed but system can continue
    CRITICAL = 50   # Critical error, system cannot function properly


class ErrorCategory(Enum):
    """Error categories for classification"""
    VALIDATION = "validation"      # Data validation errors
    CONFIGURATION = "configuration" # Configuration errors
    DATA = "data"                  # Data loading, processing errors
    EXECUTION = "execution"        # Execution errors
    INTERNAL = "internal"          # Internal system errors
    EXTERNAL = "external"          # External system/API errors
    UNKNOWN = "unknown"            # Unclassified errors


class BalanceBreakerError(Exception):
    """
    Base class for all Balance Breaker errors
    
    Attributes:
        message: Error message
        subsystem: Name of the subsystem where error occurred
        component: Name of the component where error occurred
        severity: Error severity level
        category: Error category
        timestamp: Error timestamp
        context: Additional context information
        original_exception: Original exception that caused this error
    """
    
    def __init__(self, 
                message: str, 
                subsystem: str = "",
                component: str = "",
                severity: ErrorSeverity = ErrorSeverity.ERROR,
                category: ErrorCategory = ErrorCategory.UNKNOWN,
                context: Optional[Dict[str, Any]] = None,
                original_exception: Optional[Exception] = None):
        """
        Initialize error
        
        Args:
            message: Error message
            subsystem: Name of the subsystem where error occurred
            component: Name of the component where error occurred
            severity: Error severity level
            category: Error category
            context: Additional context information
            original_exception: Original exception that caused this error
        """
        self.message = message
        self.subsystem = subsystem
        self.component = component
        self.severity = severity
        self.category = category
        self.timestamp = datetime.datetime.now()
        self.context = context or {}
        self.original_exception = original_exception
        self.traceback = traceback.format_exc() if original_exception else ""
        
        # Format the message for the base Exception class
        formatted_message = f"[{subsystem}:{component}] {message}"
        super().__init__(formatted_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary
        
        Returns:
            Dictionary representation of error
        """
        return {
            'message': self.message,
            'subsystem': self.subsystem,
            'component': self.component,
            'severity': self.severity.name,
            'category': self.category.value,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'traceback': self.traceback
        }
    
    def log(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Log the error with appropriate severity
        
        Args:
            logger: Logger to use, if None uses default logger
        """
        if logger is None:
            logger = logging.getLogger(self.subsystem or __name__)
        
        # Map severity to logging level
        level_map = {
            ErrorSeverity.DEBUG: logging.DEBUG,
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        
        level = level_map.get(self.severity, logging.ERROR)
        
        # Format error message
        log_message = f"[{self.category.value}] {self.message}"
        
        # Add context if available
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            log_message = f"{log_message} - Context: {context_str}"
        
        # Log the error
        logger.log(level, log_message)
        
        # Log traceback if available and level is high enough
        if self.traceback and level >= logging.ERROR:
            logger.log(level, f"Traceback: {self.traceback}")


# Specific error classes for different subsystems/categories

class DataPipelineError(BalanceBreakerError):
    """Error in the data pipeline subsystem"""
    
    def __init__(self, 
                message: str, 
                component: str = "",
                severity: ErrorSeverity = ErrorSeverity.ERROR,
                category: ErrorCategory = ErrorCategory.DATA,
                context: Optional[Dict[str, Any]] = None,
                original_exception: Optional[Exception] = None):
        super().__init__(
            message=message, 
            subsystem="data_pipeline",
            component=component,
            severity=severity,
            category=category,
            context=context,
            original_exception=original_exception
        )


class DataValidationError(DataPipelineError):
    """Data validation error"""
    
    def __init__(self, 
                message: str, 
                component: str = "",
                context: Optional[Dict[str, Any]] = None,
                original_exception: Optional[Exception] = None):
        super().__init__(
            message=message, 
            component=component,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.VALIDATION,
            context=context,
            original_exception=original_exception
        )


class RiskManagementError(BalanceBreakerError):
    """Error in the risk management subsystem"""
    
    def __init__(self, 
                message: str, 
                component: str = "",
                severity: ErrorSeverity = ErrorSeverity.ERROR,
                category: ErrorCategory = ErrorCategory.UNKNOWN,
                context: Optional[Dict[str, Any]] = None,
                original_exception: Optional[Exception] = None):
        super().__init__(
            message=message, 
            subsystem="risk_management",
            component=component,
            severity=severity,
            category=category,
            context=context,
            original_exception=original_exception
        )


class PortfolioError(BalanceBreakerError):
    """Error in the portfolio management subsystem"""
    
    def __init__(self, 
                message: str, 
                component: str = "",
                severity: ErrorSeverity = ErrorSeverity.ERROR,
                category: ErrorCategory = ErrorCategory.UNKNOWN,
                context: Optional[Dict[str, Any]] = None,
                original_exception: Optional[Exception] = None):
        super().__init__(
            message=message, 
            subsystem="portfolio",
            component=component,
            severity=severity,
            category=category,
            context=context,
            original_exception=original_exception
        )


class StrategyError(BalanceBreakerError):
    """Error in the strategy subsystem"""
    
    def __init__(self, 
                message: str, 
                component: str = "",
                severity: ErrorSeverity = ErrorSeverity.ERROR,
                category: ErrorCategory = ErrorCategory.UNKNOWN,
                context: Optional[Dict[str, Any]] = None,
                original_exception: Optional[Exception] = None):
        super().__init__(
            message=message, 
            subsystem="strategy",
            component=component,
            severity=severity,
            category=category,
            context=context,
            original_exception=original_exception
        )


class ConfigurationError(BalanceBreakerError):
    """Configuration error"""
    
    def __init__(self, 
                message: str, 
                subsystem: str = "",
                component: str = "",
                context: Optional[Dict[str, Any]] = None,
                original_exception: Optional[Exception] = None):
        super().__init__(
            message=message, 
            subsystem=subsystem,
            component=component,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.CONFIGURATION,
            context=context,
            original_exception=original_exception
        )


class ErrorHandler:
    """
    Error handler for standardized error management
    
    This class provides methods for handling errors consistently
    across the system.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize error handler
        
        Args:
            logger: Logger to use for error logging
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_listeners: List[callable] = []
    
    def handle_error(self, error: Union[BalanceBreakerError, Exception], 
                    context: Optional[Dict[str, Any]] = None,
                    subsystem: str = "",
                    component: str = "") -> BalanceBreakerError:
        """
        Handle an error
        
        Args:
            error: Error to handle
            context: Additional context information
            subsystem: Name of the subsystem where error occurred
            component: Name of the component where error occurred
            
        Returns:
            Standardized BalanceBreakerError
        """
        # If already a BalanceBreakerError, just log it
        if isinstance(error, BalanceBreakerError):
            bb_error = error
            
            # Add additional context if provided
            if context:
                bb_error.context.update(context)
        else:
            # Convert to BalanceBreakerError
            bb_error = BalanceBreakerError(
                message=str(error),
                subsystem=subsystem,
                component=component,
                context=context,
                original_exception=error if isinstance(error, Exception) else None
            )
        
        # Log the error
        bb_error.log(self.logger)
        
        # Notify listeners
        for listener in self.error_listeners:
            try:
                listener(bb_error)
            except Exception as e:
                self.logger.error(f"Error in error listener: {str(e)}")
        
        return bb_error
    
    def add_error_listener(self, listener: callable) -> None:
        """
        Add an error listener
        
        The listener will be called with the error object when an error is handled.
        
        Args:
            listener: Listener function that takes a BalanceBreakerError
        """
        self.error_listeners.append(listener)
    
    def remove_error_listener(self, listener: callable) -> None:
        """
        Remove an error listener
        
        Args:
            listener: Listener to remove
        """
        if listener in self.error_listeners:
            self.error_listeners.remove(listener)
    
    @contextmanager
    def error_context(self, 
                     context: Dict[str, Any],
                     subsystem: str = "",
                     component: str = "") -> None:
        """
        Context manager for handling errors with specific context
        
        Args:
            context: Context information to include in errors
            subsystem: Name of the subsystem where error occurred
            component: Name of the component where error occurred
            
        Example:
            ```
            with error_handler.error_context({'operation': 'load_data'}, 
                                           subsystem='data_pipeline'):
                data = load_data()
            ```
        """
        try:
            yield
        except Exception as e:
            self.handle_error(e, context, subsystem, component)
            raise  # Re-raise the error