"""
Interface Contract Registry

This module provides a registry of all interface contracts in the system,
enabling runtime verification of component compatibility.
"""

from typing import Dict, Any, Type, Set, List, Optional, Union, Callable
from abc import ABC, abstractmethod
import inspect
import logging

logger = logging.getLogger(__name__)

class InterfaceContract:
    """Represents an interface contract with required methods and signatures"""
    
    def __init__(self, interface_class: Type, required_methods: Set[str] = None):
        self.interface_class = interface_class
        self.name = interface_class.__name__
        self.module = interface_class.__module__
        
        # If required_methods not provided, extract all abstract methods
        if required_methods is None:
            self.required_methods = {
                name for name, method in inspect.getmembers(interface_class, predicate=inspect.isfunction)
                if getattr(method, '__isabstractmethod__', False)
            }
        else:
            self.required_methods = required_methods
        
        # Store method signatures
        self.method_signatures = {}
        for method_name in self.required_methods:
            if hasattr(interface_class, method_name):
                method = getattr(interface_class, method_name)
                self.method_signatures[method_name] = inspect.signature(method)
    
    def validate_implementation(self, implementation: Any) -> Dict[str, Any]:
        """
        Validate that a component implements this interface correctly
        
        Args:
            implementation: Component instance to validate
            
        Returns:
            Dict with validation results:
            {
                'valid': bool,
                'missing_methods': List[str],
                'incompatible_signatures': Dict[str, str]
            }
        """
        result = {
            'valid': True,
            'missing_methods': [],
            'incompatible_signatures': {}
        }
        
        # Check that implementation is an instance of interface_class
        if not isinstance(implementation, self.interface_class):
            result['valid'] = False
            result['error'] = f"Implementation is not an instance of {self.name}"
            return result
        
        # Check required methods
        for method_name in self.required_methods:
            if not hasattr(implementation, method_name) or not callable(getattr(implementation, method_name)):
                result['valid'] = False
                result['missing_methods'].append(method_name)
            else:
                # Check method signature compatibility
                impl_method = getattr(implementation, method_name)
                if method_name in self.method_signatures:
                    expected_sig = self.method_signatures[method_name]
                    actual_sig = inspect.signature(impl_method)
                    
                    # Check if signatures are compatible
                    try:
                        expected_sig.bind(*[p.name for p in actual_sig.parameters.values()])
                    except TypeError as e:
                        result['valid'] = False
                        result['incompatible_signatures'][method_name] = str(e)
        
        return result


class InterfaceRegistry:
    """Registry of all interface contracts in the system"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InterfaceRegistry, cls).__new__(cls)
            cls._instance.interfaces = {}
            cls._instance.implementations = {}
        return cls._instance
    
    def register_interface(self, interface_class: Type) -> None:
        """
        Register an interface contract
        
        Args:
            interface_class: Interface class (abstract base class)
        """
        contract = InterfaceContract(interface_class)
        self.interfaces[contract.name] = contract
        logger.debug(f"Registered interface contract: {contract.name}")
    
    def register_implementation(self, implementation_class: Type, interface_name: str) -> bool:
        """
        Register a component implementation for an interface
        
        Args:
            implementation_class: Component implementation class
            interface_name: Name of the interface it implements
            
        Returns:
            True if registration was successful, False otherwise
        """
        if interface_name not in self.interfaces:
            logger.error(f"Unknown interface: {interface_name}")
            return False
        
        if interface_name not in self.implementations:
            self.implementations[interface_name] = []
        
        # Add implementation
        self.implementations[interface_name].append(implementation_class)
        logger.debug(f"Registered implementation {implementation_class.__name__} for {interface_name}")
        return True
    
    def validate_implementation(self, implementation: Any, interface_name: str) -> Dict[str, Any]:
        """
        Validate that a component implements an interface correctly
        
        Args:
            implementation: Component instance to validate
            interface_name: Name of the interface to validate against
            
        Returns:
            Dict with validation results
        """
        if interface_name not in self.interfaces:
            return {
                'valid': False,
                'error': f"Unknown interface: {interface_name}"
            }
        
        contract = self.interfaces[interface_name]
        return contract.validate_implementation(implementation)
    
    def get_all_implementations(self, interface_name: str) -> List[Type]:
        """
        Get all registered implementations for an interface
        
        Args:
            interface_name: Name of the interface
            
        Returns:
            List of implementation classes
        """
        if interface_name not in self.implementations:
            return []
        
        return self.implementations[interface_name]
    
    def get_interface_contract(self, interface_name: str) -> Optional[InterfaceContract]:
        """
        Get an interface contract by name
        
        Args:
            interface_name: Name of the interface
            
        Returns:
            InterfaceContract or None if not found
        """
        return self.interfaces.get(interface_name)
    
    def list_interfaces(self) -> List[str]:
        """
        Get a list of all registered interfaces
        
        Returns:
            List of interface names
        """
        return list(self.interfaces.keys())
    
    def list_implementations(self, interface_name: str) -> List[str]:
        """
        Get a list of all registered implementations for an interface
        
        Args:
            interface_name: Name of the interface
            
        Returns:
            List of implementation class names
        """
        if interface_name not in self.implementations:
            return []
        
        return [impl.__name__ for impl in self.implementations[interface_name]]


# Singleton instance
registry = InterfaceRegistry()


# Decorator for interfaces
def interface(cls):
    """Decorator to register a class as an interface"""
    registry.register_interface(cls)
    return cls


# Decorator for implementations
def implements(interface_name):
    """Decorator to register a class as an implementation of an interface"""
    def decorator(cls):
        registry.register_implementation(cls, interface_name)
        return cls
    return decorator