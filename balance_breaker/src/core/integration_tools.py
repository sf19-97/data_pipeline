"""
System Integration Tools

This module provides tools for integrating the different subsystems of Balance Breaker.
It ensures consistent communication and data exchange between subsystems.
"""

from typing import Dict, Any, List, Optional, Type, Callable, Union, TypeVar, Generic
import logging
import time
import functools
import inspect
from enum import Enum

from balance_breaker.src.core.error_handling import ErrorHandler, BalanceBreakerError


# Type variables for generic types
T = TypeVar('T')
U = TypeVar('U')


class IntegrationType(Enum):
    """Types of integration between subsystems"""
    DATA_FLOW = "data_flow"  # Data is passed from one subsystem to another
    EVENT = "event"          # Event notifications between subsystems
    COMMAND = "command"      # Command execution between subsystems
    SERVICE = "service"      # Service invocation between subsystems


class Integration:
    """
    Integration definition between subsystems
    
    Defines how two subsystems communicate with each other.
    """
    
    def __init__(self, 
                source_subsystem: str,
                target_subsystem: str,
                integration_type: IntegrationType,
                description: str,
                data_schema: Optional[Dict[str, Any]] = None):
        """
        Initialize integration
        
        Args:
            source_subsystem: Source subsystem name
            target_subsystem: Target subsystem name
            integration_type: Type of integration
            description: Integration description
            data_schema: Schema of data exchanged (if applicable)
        """
        self.source_subsystem = source_subsystem
        self.target_subsystem = target_subsystem
        self.integration_type = integration_type
        self.description = description
        self.data_schema = data_schema or {}
    
    def __str__(self) -> str:
        return f"{self.source_subsystem} -> {self.target_subsystem} ({self.integration_type.value})"


class IntegrationRegistry:
    """
    Registry of all integrations between subsystems
    
    This registry tracks how subsystems are connected and what data they exchange.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IntegrationRegistry, cls).__new__(cls)
            cls._instance.integrations = []
            cls._instance.logger = logging.getLogger(__name__)
        return cls._instance
    
    def register_integration(self, integration: Integration) -> None:
        """
        Register an integration
        
        Args:
            integration: Integration to register
        """
        self.integrations.append(integration)
        self.logger.debug(f"Registered integration: {integration}")
    
    def get_integrations(self, 
                        source_subsystem: Optional[str] = None,
                        target_subsystem: Optional[str] = None,
                        integration_type: Optional[IntegrationType] = None) -> List[Integration]:
        """
        Get integrations matching criteria
        
        Args:
            source_subsystem: Filter by source subsystem
            target_subsystem: Filter by target subsystem
            integration_type: Filter by integration type
            
        Returns:
            List of matching integrations
        """
        results = self.integrations
        
        if source_subsystem:
            results = [i for i in results if i.source_subsystem == source_subsystem]
        
        if target_subsystem:
            results = [i for i in results if i.target_subsystem == target_subsystem]
        
        if integration_type:
            results = [i for i in results if i.integration_type == integration_type]
            
        return results
    
    def get_subsystem_dependencies(self, subsystem: str) -> List[str]:
        """
        Get subsystems that a subsystem depends on
        
        Args:
            subsystem: Subsystem name
            
        Returns:
            List of subsystem names that it depends on
        """
        integrations = self.get_integrations(source_subsystem=subsystem)
        return list(set(i.target_subsystem for i in integrations))
    
    def get_dependent_subsystems(self, subsystem: str) -> List[str]:
        """
        Get subsystems that depend on a subsystem
        
        Args:
            subsystem: Subsystem name
            
        Returns:
            List of subsystem names that depend on it
        """
        integrations = self.get_integrations(target_subsystem=subsystem)
        return list(set(i.source_subsystem for i in integrations))


# Singleton instance
registry = IntegrationRegistry()


# Decorator for registering integrations
def integrates_with(target_subsystem: str, 
                   integration_type: IntegrationType = IntegrationType.DATA_FLOW,
                   description: str = "",
                   data_schema: Optional[Dict[str, Any]] = None):
    """
    Decorator to register an integration with another subsystem
    
    Args:
        target_subsystem: Target subsystem name
        integration_type: Type of integration
        description: Integration description
        data_schema: Schema of data exchanged (if applicable)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        # Get source subsystem from module name
        module_parts = func.__module__.split('.')
        source_subsystem = module_parts[2] if len(module_parts) > 2 else "unknown"
        
        # Register integration
        integration = Integration(
            source_subsystem=source_subsystem,
            target_subsystem=target_subsystem,
            integration_type=integration_type,
            description=description or func.__doc__ or "",
            data_schema=data_schema
        )
        registry.register_integration(integration)
        
        # Return original function unchanged
        return func
    return decorator


class EventBus:
    """
    Event bus for subsystem communication
    
    Enables loose coupling between subsystems by allowing them to 
    communicate through events rather than direct method calls.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance.subscribers = {}
            cls._instance.logger = logging.getLogger(__name__)
            cls._instance.error_handler = ErrorHandler(cls._instance.logger)
        return cls._instance
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe to an event
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(callback)
        self.logger.debug(f"Subscribed to event: {event_type}")
    
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """
        Unsubscribe from an event
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Function to remove from subscribers
        """
        if event_type in self.subscribers and callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            self.logger.debug(f"Unsubscribed from event: {event_type}")
    
    def publish(self, event_type: str, data: Any = None) -> None:
        """
        Publish an event
        
        Args:
            event_type: Type of event to publish
            data: Event data
        """
        if event_type not in self.subscribers:
            return
        
        self.logger.debug(f"Publishing event: {event_type}")
        
        for callback in self.subscribers[event_type]:
            try:
                callback(data)
            except Exception as e:
                self.error_handler.handle_error(
                    e, 
                    context={'event_type': event_type},
                    subsystem='integration',
                    component='EventBus'
                )


# Singleton instance
event_bus = EventBus()


class ServiceRegistry:
    """
    Registry of services provided by subsystems
    
    Enables subsystems to publish services that other subsystems can consume.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceRegistry, cls).__new__(cls)
            cls._instance.services = {}
            cls._instance.logger = logging.getLogger(__name__)
        return cls._instance
    
    def register_service(self, service_name: str, 
                        provider: Any, 
                        method_name: str,
                        description: str = "") -> None:
        """
        Register a service
        
        Args:
            service_name: Name of the service
            provider: Object that provides the service
            method_name: Method name on the provider
            description: Service description
        """
        if not hasattr(provider, method_name):
            raise ValueError(f"Provider does not have method: {method_name}")
        
        self.services[service_name] = {
            'provider': provider,
            'method_name': method_name,
            'description': description or getattr(provider, method_name).__doc__ or ""
        }
        
        self.logger.debug(f"Registered service: {service_name}")
    
    def get_service(self, service_name: str) -> Optional[Callable]:
        """
        Get a service function
        
        Args:
            service_name: Name of the service
            
        Returns:
            Service function or None if not found
        """
        if service_name not in self.services:
            return None
        
        service = self.services[service_name]
        return getattr(service['provider'], service['method_name'])
    
    def call_service(self, service_name: str, *args, **kwargs) -> Any:
        """
        Call a service
        
        Args:
            service_name: Name of the service
            *args: Positional arguments to pass to the service
            **kwargs: Keyword arguments to pass to the service
            
        Returns:
            Result of the service call
            
        Raises:
            ValueError: If service not found
        """
        service_func = self.get_service(service_name)
        
        if service_func is None:
            raise ValueError(f"Service not found: {service_name}")
        
        return service_func(*args, **kwargs)
    
    def list_services(self) -> Dict[str, str]:
        """
        List all registered services
        
        Returns:
            Dictionary mapping service names to descriptions
        """
        return {
            name: svc['description'] for name, svc in self.services.items()
        }


# Singleton instance
service_registry = ServiceRegistry()


# Decorators for service registration and consumption

def provides_service(service_name: str, description: str = ""):
    """
    Decorator to register a method as a service
    
    Args:
        service_name: Name of the service
        description: Service description
        
    Returns:
        Decorated method
    """
    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            return method(self, *args, **kwargs)
        
        # Register service when the decorated method is defined
        # This requires the decorator to be used on instance methods
        service_registry.register_service(
            service_name=service_name,
            provider=self,
            method_name=method.__name__,
            description=description or method.__doc__ or ""
        )
        
        return wrapper
    return decorator


def consumes_service(service_name: str):
    """
    Decorator to inject a service dependency
    
    Args:
        service_name: Name of the service
        
    Returns:
        Decorated method
    """
    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            # Get service function
            service_func = service_registry.get_service(service_name)
            
            if service_func is None:
                raise ValueError(f"Service not found: {service_name}")
            
            # Add service to kwargs
            kwargs['service'] = service_func
            
            return method(self, *args, **kwargs)
        
        return wrapper
    return decorator


# Utilities for subsystem initialization ordering

def get_initialization_order() -> List[str]:
    """
    Get the order in which subsystems should be initialized
    
    Calculates a topological sort of subsystems based on their dependencies.
    
    Returns:
        List of subsystem names in initialization order
    """
    # Get all subsystems
    subsystems = set()
    for integration in registry.integrations:
        subsystems.add(integration.source_subsystem)
        subsystems.add(integration.target_subsystem)
    
    # Build dependency graph
    graph = {s: set(registry.get_subsystem_dependencies(s)) for s in subsystems}
    
    # Perform topological sort
    result = []
    visited = set()
    temp_visited = set()
    
    def visit(node):
        if node in temp_visited:
            # Circular dependency detected
            cycle = [node]
            return False
        if node in visited:
            return True
        
        temp_visited.add(node)
        
        for dep in graph[node]:
            if dep in subsystems:  # Only visit nodes that are actual subsystems
                if not visit(dep):
                    return False
        
        temp_visited.remove(node)
        visited.add(node)
        result.append(node)
        return True
    
    # Visit all nodes
    for subsystem in subsystems:
        if subsystem not in visited:
            if not visit(subsystem):
                # Circular dependency detected
                logging.warning("Circular dependency detected in subsystem dependencies")
                break
    
    # Reverse result to get initialization order
    return list(reversed(result))