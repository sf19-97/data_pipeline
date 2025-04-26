# Balance Breaker Core Implementation Guide

This guide explains how to use the core framework components to implement consistent and robust subsystems. It covers the use of core interfaces, data models, parameter management, error handling, and integration tools.

## Table of Contents

1. [Interface Contracts](#interface-contracts)
2. [Data Models](#data-models)
3. [Parameter Management](#parameter-management)
4. [Error Handling](#error-handling)
5. [Subsystem Integration](#subsystem-integration)
6. [Implementation Examples](#implementation-examples)

## Interface Contracts

The `interface_registry.py` module provides tools for defining and enforcing interface contracts between components.

### Defining an Interface

```python
from balance_breaker.src.core.interface_registry import interface

@interface
class DataProcessor(ABC):
    """Interface for data processing components"""
    
    @abstractmethod
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process data according to component logic"""
        pass
```

### Implementing an Interface

```python
from balance_breaker.src.core.interface_registry import implements

@implements("DataProcessor")
class NormalizationProcessor(DataProcessor):
    """Normalization processor implementation"""
    
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        # Implementation
        return normalized_data
```

### Validating Implementations

```python
from balance_breaker.src.core.interface_registry import registry

# Validate an implementation
processor = NormalizationProcessor()
validation = registry.validate_implementation(processor, "DataProcessor")

if not validation['valid']:
    print(f"Validation failed: {validation}")
```

## Data Models

The `data_models.py` module defines core data models that are shared across subsystems.

### Using Data Models

```python
from balance_breaker.src.core.data_models import TradeParameters, Direction

# Create a trade parameters instance
trade_params = TradeParameters(
    instrument="EURUSD",
    direction=Direction.LONG,
    entry_price=1.1250,
    stop_loss=1.1200,
    take_profit=[1.1300, 1.1350],
    position_size=1.0,
    risk_amount=50.0,
    risk_percent=0.02
)

# Access fields
print(f"Position size: {trade_params.position_size}")
```

### Converting Models to Different Formats

```python
from balance_breaker.src.core.data_models import model_to_dict, model_to_json

# Convert to dictionary
params_dict = model_to_dict(trade_params)

# Convert to JSON
params_json = model_to_json(trade_params)
```

## Parameter Management

The `parameter_manager.py` module provides tools for managing component parameters.

### Creating a Parameterized Component

```python
from balance_breaker.src.core.parameter_manager import ParameterizedComponent

class MyComponent(ParameterizedComponent):
    """Example component with parameterized configuration"""
    
    def __init__(self, parameters=None):
        """
        Initialize component
        
        Parameters:
        -----------
        param1 : int
            First parameter (default: 10)
        param2 : float
            Second parameter (default: 1.5)
        """
        # Define default parameters
        default_params = {
            'param1': 10,
            'param2': 1.5
        }
        
        # Update defaults with provided parameters
        if parameters:
            default_params.update(parameters)
            
        super().__init__(default_params)
    
    def process(self, data):
        # Use parameters
        value1 = self.parameters['param1']
        value2 = self.parameters['param2']
        
        # Process logic
        return data * value1 + value2
```

### Defining Parameter Schema

```python
from balance_breaker.src.core.parameter_manager import (
    ParameterManager, ParameterSchema, ParameterDefinition, ParameterType
)

# Define schema manually
schema = ParameterSchema(
    parameters={
        'param1': ParameterDefinition(
            name='param1',
            parameter_type=ParameterType.INTEGER,
            default_value=10,
            description='First parameter',
            minimum=1,
            maximum=100
        ),
        'param2': ParameterDefinition(
            name='param2',
            parameter_type=ParameterType.FLOAT,
            default_value=1.5,
            description='Second parameter',
            minimum=0.0,
            maximum=10.0
        )
    },
    component_name='MyComponent',
    description='Example component with parameterized configuration'
)

# Create parameter manager with schema
param_manager = ParameterManager(schema)

# Validate parameters
params = {'param1': 5, 'param2': 2.5}
issues = param_manager.validate_parameters(params)

if issues:
    print(f"Validation issues: {issues}")
else:
    print("Parameters valid")
```

## Error Handling

The `error_handling.py` module provides standardized error handling across subsystems.

### Defining Custom Errors

```python
from balance_breaker.src.core.error_handling import (
    BalanceBreakerError, ErrorSeverity, ErrorCategory
)

class MyComponentError(BalanceBreakerError):
    """Error in my component"""
    
    def __init__(self, 
                message: str, 
                component: str = "",
                severity: ErrorSeverity = ErrorSeverity.ERROR,
                context: Optional[Dict[str, Any]] = None,
                original_exception: Optional[Exception] = None):
        super().__init__(
            message=message, 
            subsystem="my_subsystem",
            component=component,
            severity=severity,
            category=ErrorCategory.EXECUTION,
            context=context,
            original_exception=original_exception
        )
```

### Using the Error Handler

```python
from balance_breaker.src.core.error_handling import ErrorHandler

# Create error handler
error_handler = ErrorHandler()

try:
    # Some operation that might fail
    result = process_data(data)
except Exception as e:
    # Handle the error
    bb_error = error_handler.handle_error(
        e,
        context={'operation': 'process_data', 'data_size': len(data)},
        subsystem='data_pipeline',
        component='DataProcessor'
    )
    
    # Log already happens in handle_error
    
    # You can access error details
    error_dict = bb_error.to_dict()
    print(f"Error severity: {bb_error.severity.name}")
```

### Using Error Context

```python
# Create context manager for error handling
with error_handler.error_context(
    context={'operation': 'load_data'},
    subsystem='data_pipeline',
    component='DataLoader'
):
    # Code that might raise an exception
    data = load_data(file_path)
```

## Subsystem Integration

The `integration_tools.py` module provides tools for integrating subsystems.

### Registering Integrations

```python
from balance_breaker.src.core.integration_tools import (
    integrates_with, IntegrationType
)

class DataPipeline:
    @integrates_with(
        target_subsystem='portfolio',
        integration_type=IntegrationType.DATA_FLOW,
        description='Provides processed price and macro data to portfolio'
    )
    def get_data(self, pairs, start_date, end_date):
        # Implementation
        return processed_data
```

### Using the Event Bus

```python
from balance_breaker.src.core.integration_tools import event_bus

# Subscribe to an event
def handle_position_closed(data):
    print(f"Position closed: {data['instrument']}")

event_bus.subscribe('position_closed', handle_position_closed)

# Publish an event
event_bus.publish('position_closed', {
    'instrument': 'EURUSD',
    'profit': 100.0,
    'time': '2023-01-15T12:34:56'
})
```

### Using the Service Registry

```python
from balance_breaker.src.core.integration_tools import (
    service_registry, provides_service, consumes_service
)

class RiskManager:
    @provides_service('calculate_position_size')
    def calculate_position_size(self, balance, risk_percent, stop_distance):
        # Implementation
        return position_size

class Portfolio:
    @consumes_service('calculate_position_size')
    def open_position(self, instrument, price, service=None):
        # Use the injected service
        position_size = service(
            balance=self.balance,
            risk_percent=0.02,
            stop_distance=50
        )
        
        # Implementation
        return position_id
```

## Implementation Examples

### Basic Component Implementation

```python
from balance_breaker.src.core.interface_registry import interface, implements
from balance_breaker.src.core.parameter_manager import ParameterizedComponent
from balance_breaker.src.core.error_handling import ErrorHandler, ErrorCategory, ErrorSeverity

# Define the interface
@interface
class Processor(ABC):
    @abstractmethod
    def process(self, data, context):
        pass

# Implement the interface
@implements("Processor")
class SimpleProcessor(ParameterizedComponent):
    """
    Simple data processor
    
    Parameters:
    -----------
    multiplier : float
        Value to multiply data by (default: 1.0)
    offset : float
        Value to add to data (default: 0.0)
    """
    
    def __init__(self, parameters=None):
        default_params = {
            'multiplier': 1.0,
            'offset': 0.0
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
        
        # Create error handler
        self.error_handler = ErrorHandler()
    
    def process(self, data, context):
        try:
            # Get parameters
            multiplier = self.parameters['multiplier']
            offset = self.parameters['offset']
            
            # Process data
            result = data * multiplier + offset
            
            return result
            
        except Exception as e:
            # Handle errors
            self.error_handler.handle_error(
                e,
                context={'data_shape': getattr(data, 'shape', None)},
                subsystem='data_pipeline',
                component='SimpleProcessor'
            )
            raise  # Re-raise after handling
```

### Orchestrator Implementation

```python
from balance_breaker.src.core.interface_registry import registry
from balance_breaker.src.core.error_handling import ErrorHandler
from balance_breaker.src.core.integration_tools import integrates_with, IntegrationType

class SimpleOrchestrator:
    """Simple orchestrator for demonstration"""
    
    def __init__(self):
        self.components = {}
        self.error_handler = ErrorHandler()
    
    def register_component(self, name, component):
        # Validate component implements required interface
        validation = registry.validate_implementation(component, "Processor")
        
        if not validation['valid']:
            raise ValueError(f"Component validation failed: {validation}")
        
        self.components[name] = component
    
    def create_pipeline(self, component_names):
        # Create pipeline from registered components
        pipeline = []
        
        for name in component_names:
            if name not in self.components:
                raise ValueError(f"Component not found: {name}")
            
            pipeline.append(self.components[name])
        
        return pipeline
    
    @integrates_with(
        target_subsystem='portfolio',
        integration_type=IntegrationType.DATA_FLOW,
        description='Processes data for portfolio system'
    )
    def execute_pipeline(self, pipeline, data, context):
        result = data
        
        for component in pipeline:
            try:
                result = component.process(result, context)
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context={'component': component.__class__.__name__},
                    subsystem='data_pipeline',
                    component='SimpleOrchestrator'
                )
                raise
        
        return result
```

### Advanced Integration Example

```python
from balance_breaker.src.core.data_models import Portfolio, TradeParameters
from balance_breaker.src.core.integration_tools import service_registry, event_bus

class PortfolioManager:
    def __init__(self):
        self.portfolio = Portfolio(
            name="Main Portfolio",
            base_currency="USD",
            initial_capital=100000.0
        )
        
        # Register services
        service_registry.register_service(
            service_name='get_portfolio_state',
            provider=self,
            method_name='get_portfolio',
            description='Get current portfolio state'
        )
    
    def get_portfolio(self):
        return self.portfolio
    
    def open_position(self, trade_params: TradeParameters):
        # Implementation to open a position
        position_id = "pos_" + uuid.uuid4().hex[:8]
        
        # Create position
        position = PortfolioPosition(
            instrument=trade_params.instrument,
            direction=trade_params.direction,
            entry_price=trade_params.entry_price,
            position_size=trade_params.position_size,
            stop_loss=trade_params.stop_loss,
            take_profit=trade_params.take_profit,
            position_id=position_id,
            risk_amount=trade_params.risk_amount,
            risk_percent=trade_params.risk_percent
        )
        
        # Add to portfolio
        self.portfolio.positions[trade_params.instrument] = position
        
        # Publish event
        event_bus.publish('position_opened', {
            'instrument': trade_params.instrument,
            'direction': trade_params.direction.value,
            'size': trade_params.position_size,
            'position_id': position_id
        })
        
        return position_id
```

By following these patterns, you can ensure consistency across all subsystems while minimizing coupling and maximizing reusability.