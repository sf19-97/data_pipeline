import inspect
"""
Parameter Management System

This module provides a standardized way to manage component parameters
across all subsystems of Balance Breaker.
"""

from typing import Dict, Any, List, Optional, Set, Type, TypeVar, Generic, Union, get_type_hints
import json
import os
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ParameterType(Enum):
    """Parameter types for validation and UI rendering"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ENUM = "enum"
    LIST = "list"
    DICT = "dict"
    PATH = "path"


@dataclass
class ParameterDefinition:
    """Definition of a parameter with metadata for validation and UI rendering"""
    name: str
    parameter_type: ParameterType
    default_value: Any
    description: str = ""
    required: bool = False
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[List[Any]] = None
    multiline: bool = False
    advanced: bool = False
    category: str = "General"
    dependent_on: Optional[Dict[str, Any]] = None


@dataclass
class ParameterSchema:
    """Schema for a set of parameters"""
    parameters: Dict[str, ParameterDefinition]
    schema_version: str = "1.0.0"
    component_name: str = ""
    component_type: str = ""
    description: str = ""


class ParameterManager:
    """
    Manager for component parameters
    
    Provides parameter validation, default value handling, and
    schema-based operations.
    """
    
    def __init__(self, schema: Optional[ParameterSchema] = None):
        """
        Initialize with optional schema
        
        Args:
            schema: Parameter schema
        """
        self.schema = schema or ParameterSchema({})
        
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters against schema
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Dictionary of validation issues by parameter name.
            Empty dict if all valid.
        """
        if not self.schema or not self.schema.parameters:
            return {}  # No schema, assume valid
        
        issues = {}
        
        # Check each parameter against its definition
        for name, definition in self.schema.parameters.items():
            # Skip if not required and not provided
            if not definition.required and name not in parameters:
                continue
                
            # Check if required parameter is missing
            if definition.required and name not in parameters:
                issues[name] = f"Required parameter '{name}' is missing"
                continue
                
            # If parameter is provided, validate it
            if name in parameters:
                value = parameters[name]
                param_type = definition.parameter_type
                
                # Type validation
                if param_type == ParameterType.STRING and not isinstance(value, str):
                    issues[name] = f"Expected string, got {type(value).__name__}"
                
                elif param_type == ParameterType.INTEGER:
                    if not isinstance(value, int) or isinstance(value, bool):
                        issues[name] = f"Expected integer, got {type(value).__name__}"
                    elif definition.minimum is not None and value < definition.minimum:
                        issues[name] = f"Value {value} is less than minimum {definition.minimum}"
                    elif definition.maximum is not None and value > definition.maximum:
                        issues[name] = f"Value {value} is greater than maximum {definition.maximum}"
                
                elif param_type == ParameterType.FLOAT:
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        issues[name] = f"Expected number, got {type(value).__name__}"
                    elif definition.minimum is not None and value < definition.minimum:
                        issues[name] = f"Value {value} is less than minimum {definition.minimum}"
                    elif definition.maximum is not None and value > definition.maximum:
                        issues[name] = f"Value {value} is greater than maximum {definition.maximum}"
                
                elif param_type == ParameterType.BOOLEAN and not isinstance(value, bool):
                    issues[name] = f"Expected boolean, got {type(value).__name__}"
                
                elif param_type == ParameterType.ENUM:
                    if definition.choices is None:
                        issues[name] = "Enum parameter definition missing 'choices'"
                    elif value not in definition.choices:
                        issues[name] = f"Value {value} not in choices: {definition.choices}"
                
                elif param_type == ParameterType.LIST and not isinstance(value, list):
                    issues[name] = f"Expected list, got {type(value).__name__}"
                
                elif param_type == ParameterType.DICT and not isinstance(value, dict):
                    issues[name] = f"Expected dict, got {type(value).__name__}"
                
                elif param_type == ParameterType.PATH:
                    if not isinstance(value, str):
                        issues[name] = f"Expected path string, got {type(value).__name__}"
                    elif not os.path.exists(value):
                        issues[name] = f"Path '{value}' does not exist"
        
        return issues
    
    def apply_defaults(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply default values for missing parameters
        
        Args:
            parameters: Input parameters
            
        Returns:
            Parameters with defaults applied
        """
        if not self.schema or not self.schema.parameters:
            return parameters.copy()
        
        result = parameters.copy()
        
        # Apply defaults for missing parameters
        for name, definition in self.schema.parameters.items():
            if name not in result:
                result[name] = definition.default_value
        
        return result
    
    def filter_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter parameters to include only those defined in schema
        
        Args:
            parameters: Input parameters
            
        Returns:
            Filtered parameters
        """
        if not self.schema or not self.schema.parameters:
            return parameters.copy()
        
        return {
            key: value for key, value in parameters.items()
            if key in self.schema.parameters
        }
    
    @staticmethod
    def create_schema_from_class(cls: Type[Any]) -> ParameterSchema:
        """
        Create parameter schema from class type hints and docstring
        
        Args:
            cls: Class with type hints and parameter documentation
            
        Returns:
            Parameter schema
        """
        # Get class name and docstring
        class_name = cls.__name__
        doc = cls.__doc__ or ""
        
        # Get type hints
        hints = get_type_hints(cls)
        
        # Check for __init__ method with parameters
        init_params = {}
        if hasattr(cls, '__init__'):
            init = getattr(cls, '__init__')
            if callable(init):
                # Get init parameters
                sig = inspect.signature(init)
                init_params = {
                    name: param.default if param.default is not inspect.Parameter.empty else None
                    for name, param in sig.parameters.items()
                    if name != 'self' and name != 'parameters'  # Skip self and parameters dict
                }
        
        # Parse docstring for parameter descriptions
        param_descriptions = {}
        if doc:
            param_section = False
            current_param = None
            desc_lines = []
            
            for line in doc.split("\n"):
                line = line.strip()
                
                # Check if we're in a Parameters section
                if line.lower().startswith("parameters:"):
                    param_section = True
                    continue
                
                # Check if we're exiting the Parameters section
                if param_section and line and line[0].isalpha() and line.endswith(":"):
                    param_section = False
                    
                # Process parameter description
                if param_section:
                    if line and (line[0] == "-" or (line[0].isalpha() and ":" in line)):
                        # Save previous parameter description
                        if current_param:
                            param_descriptions[current_param] = " ".join(desc_lines).strip()
                            desc_lines = []
                        
                        # Extract new parameter name
                        if ":" in line:
                            parts = line.split(":", 1)
                            current_param = parts[0].strip()
                            if current_param:
                                desc_part = parts[1].strip()
                                if desc_part:
                                    desc_lines.append(desc_part)
                        else:
                            # Handle "- param_name" format
                            parts = line.split(" ", 1)
                            if len(parts) > 1:
                                current_param = parts[1].strip().split()[0]  # Get first word after dash
                    
                    elif current_param and line:
                        desc_lines.append(line)
            
            # Save last parameter description
            if current_param and desc_lines:
                param_descriptions[current_param] = " ".join(desc_lines).strip()
        
        # Create parameter definitions
        parameters = {}
        
        # Start with init parameters if available
        for name, default in init_params.items():
            param_type = ParameterType.STRING  # Default
            
            # Determine parameter type from default value
            if isinstance(default, bool):
                param_type = ParameterType.BOOLEAN
            elif isinstance(default, int):
                param_type = ParameterType.INTEGER
            elif isinstance(default, float):
                param_type = ParameterType.FLOAT
            elif isinstance(default, list):
                param_type = ParameterType.LIST
            elif isinstance(default, dict):
                param_type = ParameterType.DICT
            
            # Get description from docstring
            description = param_descriptions.get(name, "")
            
            # Create parameter definition
            parameters[name] = ParameterDefinition(
                name=name,
                parameter_type=param_type,
                default_value=default,
                description=description,
                required=default is None  # Consider required if no default
            )
        
        # Create schema
        return ParameterSchema(
            parameters=parameters,
            component_name=class_name,
            description=doc.split("\n\n")[0] if doc else ""
        )
    
    @staticmethod
    def load_schema_from_file(file_path: str) -> ParameterSchema:
        """
        Load parameter schema from a JSON file
        
        Args:
            file_path: Path to JSON schema file
            
        Returns:
            Parameter schema
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert raw data to ParameterSchema
            parameters = {}
            for name, param_data in data.get('parameters', {}).items():
                # Convert string parameter_type to enum
                param_type = ParameterType[param_data.get('parameter_type', 'STRING').upper()]
                
                # Create parameter definition
                parameters[name] = ParameterDefinition(
                    name=name,
                    parameter_type=param_type,
                    default_value=param_data.get('default_value'),
                    description=param_data.get('description', ''),
                    required=param_data.get('required', False),
                    minimum=param_data.get('minimum'),
                    maximum=param_data.get('maximum'),
                    choices=param_data.get('choices'),
                    multiline=param_data.get('multiline', False),
                    advanced=param_data.get('advanced', False),
                    category=param_data.get('category', 'General'),
                    dependent_on=param_data.get('dependent_on')
                )
            
            return ParameterSchema(
                parameters=parameters,
                schema_version=data.get('schema_version', '1.0.0'),
                component_name=data.get('component_name', ''),
                component_type=data.get('component_type', ''),
                description=data.get('description', '')
            )
            
        except Exception as e:
            logger.error(f"Error loading schema from {file_path}: {str(e)}")
            return ParameterSchema({})
    
    def save_schema_to_file(self, file_path: str) -> bool:
        """
        Save parameter schema to a JSON file
        
        Args:
            file_path: Path to save JSON schema
            
        Returns:
            True if successful, False otherwise
        """
        if not self.schema:
            logger.error("No schema to save")
            return False
        
        try:
            # Convert schema to dict
            data = {
                'schema_version': self.schema.schema_version,
                'component_name': self.schema.component_name,
                'component_type': self.schema.component_type,
                'description': self.schema.description,
                'parameters': {}
            }
            
            # Convert parameters
            for name, param in self.schema.parameters.items():
                data['parameters'][name] = {
                    'parameter_type': param.parameter_type.value,
                    'default_value': param.default_value,
                    'description': param.description,
                    'required': param.required
                }
                
                # Add optional fields if present
                if param.minimum is not None:
                    data['parameters'][name]['minimum'] = param.minimum
                
                if param.maximum is not None:
                    data['parameters'][name]['maximum'] = param.maximum
                
                if param.choices is not None:
                    data['parameters'][name]['choices'] = param.choices
                
                if param.multiline:
                    data['parameters'][name]['multiline'] = param.multiline
                
                if param.advanced:
                    data['parameters'][name]['advanced'] = param.advanced
                
                if param.category != 'General':
                    data['parameters'][name]['category'] = param.category
                
                if param.dependent_on is not None:
                    data['parameters'][name]['dependent_on'] = param.dependent_on
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving schema to {file_path}: {str(e)}")
            return False


class ParameterizedComponent:
    """
    Mixin for components that use parameterized configuration
    
    Provides standardized parameter handling with validation
    and defaults.
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize with parameters
        
        Args:
            parameters: Component parameters
        """
        # Create schema if not already created
        if not hasattr(self, '_param_schema'):
            self._param_schema = ParameterManager.create_schema_from_class(self.__class__)
        
        # Create parameter manager
        self._param_manager = ParameterManager(self._param_schema)
        
        # Initialize parameters
        self._initialize_parameters(parameters)
    
    def _initialize_parameters(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize parameters with validation and defaults
        
        Args:
            parameters: Initial parameters
        """
        # Start with empty dict if none provided
        params = parameters or {}
        
        # Apply defaults
        params = self._param_manager.apply_defaults(params)
        
        # Validate parameters
        issues = self._param_manager.validate_parameters(params)
        if issues:
            issue_str = "; ".join([f"{k}: {v}" for k, v in issues.items()])
            logger.warning(f"Parameter validation issues for {self.__class__.__name__}: {issue_str}")
        
        # Store parameters
        self.parameters = params
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        return self.parameters.copy()
    
    def set_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update parameters with validation
        
        Args:
            parameters: New parameters
            
        Returns:
            Dictionary of validation issues (empty if none)
        """
        if not parameters:
            return {}
        
        # Validate new parameters
        issues = self._param_manager.validate_parameters(parameters)
        
        # Update parameters
        self.parameters.update(parameters)
        
        return issues
    
    def get_parameter_schema(self) -> ParameterSchema:
        """Get parameter schema"""
        return self._param_schema
    
    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """Get default parameters for this component"""
        schema = ParameterManager.create_schema_from_class(cls)
        return {
            name: param.default_value 
            for name, param in schema.parameters.items()
        }