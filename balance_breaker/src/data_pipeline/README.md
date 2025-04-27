```markdown
# Balance Breaker Data Pipeline

The Balance Breaker Data Pipeline is a modular system for loading, processing, and managing financial data for use in trading strategies. This guide covers the essential concepts and provides code examples to get you started.

## Architecture Overview

The pipeline follows a registry-based architecture:

1. **Data Registry**: Central storage for data products with lifecycle management
2. **Data Orchestrator**: Manages component registration and pipeline execution
3. **Component Types**:
   - **Loaders**: Load data from repositories
   - **Validators**: Check data quality
   - **Processors**: Normalize and transform data
   - **Aligners**: Synchronize different time series
   - **Indicators**: Calculate derived metrics
   - **Serializers**: Save and cache data

## Getting Started

### Basic Usage

```python
from balance_breaker.src.data_pipeline import (
    DataPipelineOrchestrator, InMemoryDataRegistry,
    PriceLoader, MacroLoader, DataValidator,
    DataNormalizer, TimeAligner
)

# Create and configure orchestrator
orchestrator = DataPipelineOrchestrator({
    'auto_register_data': True,
    'pipeline_config_path': 'config/pipelines.json'
})

# Create and set the registry
registry = InMemoryDataRegistry()
orchestrator.set_registry(registry)

# Register components
orchestrator.register_loader(PriceLoader())
orchestrator.register_loader(MacroLoader())
orchestrator.register_validator(DataValidator())
orchestrator.register_processor(DataNormalizer())
orchestrator.register_aligner(TimeAligner())

# Register modular indicators
orchestrator.register_modular_indicators()

# Option 1: Create pipeline with configuration
pipeline_config = {
    'id': 'price_analysis',
    'loaders': ['PriceLoader'],
    'validators': ['DataValidator'],
    'processors': ['DataNormalizer'],
    'aligners': ['TimeAligner'],
    'indicators': ['RSIIndicator', 'MovingAverageIndicator', 'MACDIndicator']
}

pipeline_id = orchestrator.create_pipeline(pipeline_config)

# Execute pipeline with context
context = {
    'pairs': ['EURUSD', 'USDJPY'],
    'start_date': '2023-01-01',
    'end_date': '2023-12-31',
    'timeframe': '1h'
}

result = orchestrator.execute_pipeline(pipeline_id, context)

# Option 2: Legacy request-based API (maintained for backward compatibility)
request = {
    'pairs': ['USDJPY'],
    'start_date': '2022-01-01',
    'end_date': '2023-01-01',
    'data_type': 'price',
    'align': True,
    'indicators': ['rsi', 'ma', 'macd']
}

result = orchestrator.get_data(**request)
```

### Loading Data for Backtesting

```python
def load_data_for_backtest(
    orchestrator,
    pairs,
    start_date,
    end_date=None,
    timeframe='1d',
    indicators=None
):
    """Load and process data for backtesting"""
    # Create pipeline configuration
    pipeline_config = {
        'id': f'backtest_{timeframe}',
        'loaders': ['PriceLoader'],
        'validators': ['DataValidator', 'GapDetector'],
        'processors': ['DataNormalizer'],
        'aligners': ['TimeAligner'],
        'indicators': indicators or ['RSIIndicator', 'MovingAverageIndicator', 'MACDIndicator', 'VolatilityIndicator']
    }
    
    # Create pipeline
    pipeline_id = orchestrator.create_pipeline(pipeline_config)
    
    # Create context
    context = {
        'pairs': pairs,
        'start_date': start_date,
        'end_date': end_date,
        'timeframe': timeframe,
        'detect_gaps': True
    }
    
    # Execute pipeline
    return orchestrator.execute_pipeline(pipeline_id, context)
```

## Data Registry

The Data Registry manages the lifecycle of data products:

```python
# Register data with the registry
data_id = registry.register_data(
    data={'price': price_data},
    metadata={
        'data_type': 'price',
        'pairs': ['EURUSD'],
        'timeframe': '1h',
        'tags': ['backtest', 'analysis']
    }
)

# Retrieve data by ID
data = registry.get_data(data_id)

# Find data by criteria
results = registry.find_data({
    'data_type': 'price',
    'tags': 'analysis',
    'after_date': datetime(2023, 1, 1)
})

# Create relationships between data products
registry.create_relationship(
    source_id=raw_data_id,
    target_id=processed_data_id,
    relationship_type='derived_from'
)

# Get related data
related_ids = registry.get_related_data(
    data_id=processed_data_id,
    relationship_type='derived_from'
)
```

## Component Guide

### Data Loaders

The loaders are responsible for reading data from repositories:

- **PriceLoader**: Loads price data for currency pairs
- **MacroLoader**: Loads macroeconomic indicators
- **CustomLoader**: Flexible loader for custom data sources

```python
# Configure price loader
price_loader = PriceLoader({
    'repository_path': '/path/to/price/data',
    'default_timeframe': '1h'
})
orchestrator.register_loader(price_loader)
```

### Validators

Validators check data quality and report issues:

- **DataValidator**: Validates price and macro data quality
- **GapDetector**: Detects and reports time gaps in data
- **QualityChecker**: Checks for data quality issues

```python
# Register validator
gap_detector = GapDetector({
    'max_gap_threshold': 3,  # Max number of missing bars
    'report_only': False     # Fail if gaps exceed threshold
})
orchestrator.register_validator(gap_detector)
```

### Processors

Processors transform data into a standardized format:

- **DataNormalizer**: Normalizes price and macro data format
- **FeatureCreator**: Creates trading features from data
- **DataTransformer**: Applies transformations to data

```python
# Create processor
normalizer = DataNormalizer({
    'column_mapping': {
        'bid_close': 'close',
        'bid_high': 'high'
    }
})
orchestrator.register_processor(normalizer)
```

### Aligners

Aligners synchronize different time series:

- **TimeAligner**: Aligns price and macro data
- **TimeResampler**: Changes data timeframe (e.g., 1H → 4H)

```python
# Configure aligner
resampler = TimeResampler({
    'default_method': 'last',
    'volume_method': 'sum'
})
orchestrator.register_aligner(resampler)
```

### Indicators

Balance Breaker supports both monolithic and modular indicator systems:

#### Modular Indicators (Recommended)

Individual, focused classes for specific indicators:

- **RSIIndicator**: Relative Strength Index
- **MovingAverageIndicator**: Simple and exponential moving averages
- **MACDIndicator**: Moving Average Convergence Divergence
- **MomentumIndicator**: Price momentum and rate of change
- **VolatilityIndicator**: Price volatility measures

```python
# Register all modular indicators
orchestrator.register_modular_indicators()

# Register specific indicator with custom parameters
rsi = RSIIndicator({'period': 21, 'price_column': 'close'})
orchestrator.register_indicator(rsi)
```

#### Original Monolithic Indicators (Legacy)

Comprehensive classes that calculate multiple indicators:

- **TechnicalIndicators**: All technical indicators in one class
- **EconomicIndicators**: Economic indicators (yield spreads, inflation differentials)
- **CompositeIndicators**: Balance Breaker specific indicators (precession, market mood)

```python
# Register monolithic indicator
tech_indicators = TechnicalIndicators({
    'sma_periods': [10, 20, 50, 200],
    'rsi_period': 14
})
orchestrator.register_indicator(tech_indicators)
```

### Serializers

Serializers save and cache data:

- **DataExporter**: Exports data to various formats
- **CacheManager**: Caches pipeline results

```python
# Configure exporter
exporter = DataExporter({
    'default_format': 'csv',
    'export_dir': 'exported_data'
})
orchestrator.register_serializer(exporter)
```

## Configuration-Driven Pipelines

The new architecture supports configuration-driven pipeline creation:

```python
# Pipelines can be defined in JSON
pipeline_configs = [
    {
        "id": "price_analysis",
        "loaders": ["PriceLoader"],
        "validators": ["DataValidator"],
        "processors": ["DataNormalizer"],
        "aligners": ["TimeAligner"],
        "indicators": ["RSIIndicator", "MACDIndicator"],
        "auto_register_data": true,
        "data_types": ["price"]
    },
    {
        "id": "macro_analysis",
        "loaders": ["MacroLoader"],
        "validators": ["DataValidator"],
        "processors": ["DataNormalizer"],
        "indicators": ["EconomicIndicators"],
        "auto_execute": true
    }
]

# Load from file
orchestrator = DataPipelineOrchestrator({
    'pipeline_config_path': 'config/pipelines.json'
})

# Or create programmatically
for config in pipeline_configs:
    orchestrator.create_pipeline(config)
```

## Integration with Signal Subsystem

The Data Pipeline integrates with the Signal subsystem:

```python
# Get data formatted for signal generation
data = orchestrator.get_data_for_signals({
    'pairs': ['EURUSD'],
    'timeframe': '1h',
    'start_date': '2023-01-01',
    'end_date': '2023-12-31',
    'indicators': ['rsi', 'macd', 'volatility']
})

# Use in signal orchestrator
from balance_breaker.src.signal import SignalOrchestrator

signal_orchestrator = SignalOrchestrator()
signals = signal_orchestrator.generate_signals_from_data(
    data, 
    {'timeframe': '1h', 'generators': ['TechnicalSignalGenerator']}
)
```

## Working with Results

The pipeline typically returns a dictionary with:

- **price**: Price data by pair
- **aligned_macro**: Macro data aligned to price data timeframe

```python
# Access processed data
for pair, price_df in result['price'].items():
    print(f"Price data for {pair}: {len(price_df)} rows")
    
    # Print technical indicator columns
    tech_cols = [col for col in price_df.columns if col in ['RSI', 'SMA_20', 'MACD']]
    if tech_cols:
        print(f"Technical indicators: {', '.join(tech_cols)}")
    
    # Access aligned macro data
    if pair in result['aligned_macro']:
        macro_df = result['aligned_macro'][pair]
        print(f"Macro data for {pair}: {len(macro_df)} rows")
```

## Advanced Usage

### Event-Based Processing

The pipeline supports event-based data processing:

```python
# New data will trigger automatic pipeline execution
event_bus.publish('new_data_available', {
    'data_type': 'price',
    'pairs': ['EURUSD'],
    'timeframe': '1h'
})

# Subscribe to pipeline execution events
def on_pipeline_executed(event_data):
    pipeline_id = event_data['pipeline_id']
    print(f"Pipeline {pipeline_id} executed in {event_data['duration']}s")

event_bus.subscribe('pipeline_executed', on_pipeline_executed)
```

### Creating Custom Components

You can create custom components by implementing the interfaces:

```python
from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.core.parameter_manager import ParameterizedComponent
from balance_breaker.src.data_pipeline.data_pipeline_interfaces import DataProcessor

@implements("DataProcessor")
class MyCustomProcessor(ParameterizedComponent, DataProcessor):
    """Custom data processor implementation"""
    
    def __init__(self, parameters=None):
        default_params = {
            'option1': 'default_value',
            'option2': 42
        }
        super().__init__(parameters or default_params)
    
    @property
    def component_type(self) -> str:
        return 'processor'
    
    def process_data(self, data, context):
        # Custom processing logic
        return processed_data
```

## Best Practices

1. **Use the Data Registry**: Store and retrieve data through the registry
2. **Create configuration-driven pipelines**: Define pipelines in configuration
3. **Prefer modular indicators**: Use focused indicators for better maintainability
4. **Set appropriate context**: Provide all necessary information in the context
5. **Handle errors**: Use error handling for robust pipelines
6. **Enable caching**: Use the registry's caching capabilities
7. **Monitor performance**: Use `get_performance_metrics()` to identify bottlenecks
```