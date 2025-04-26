# Balance Breaker Data Pipeline Quick Start Guide

The Balance Breaker Data Pipeline is a modular system for loading, processing, and aligning financial data for use in trading strategies. This guide covers the essential concepts and provides code examples to get you started.

## Architecture Overview

The pipeline follows an orchestrator-based architecture:

1. **Data Orchestrator**: Central coordinator that manages the flow of data through various components
2. **Component Types**:
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
    DataPipelineOrchestrator,
    PriceLoader,
    MacroLoader,
    DataValidator,
    DataNormalizer,
    TimeAligner
)

# Create and configure orchestrator
orchestrator = DataPipelineOrchestrator()

# Register components
orchestrator.register_component(PriceLoader())
orchestrator.register_component(MacroLoader())
orchestrator.register_component(DataValidator())
orchestrator.register_component(DataNormalizer())
orchestrator.register_component(TimeAligner())

# Register modular indicators (auto-discovers all available indicators)
orchestrator.register_modular_indicators()

# Define request
request = {
    'pairs': ['USDJPY'],
    'start_date': '2022-01-01',
    'end_date': '2023-01-01',
    'data_type': 'price',
    'align': True,
    'indicators': ['rsi', 'ma', 'macd']  # Request specific indicators by name
}

# Create and execute pipeline
pipeline = orchestrator.create_pipeline(request)
result = orchestrator.execute_pipeline(pipeline, request)
```

### Loading Data for Backtesting

```python
def load_data_for_backtest(
    orchestrator,
    pairs,
    start_date,
    end_date=None,
    repository_config=None,
    indicators=None
):
    """Load and process data for backtesting"""
    # Create request context
    context = {
        'pairs': pairs,
        'start_date': start_date,
        'end_date': end_date,
        'repository_config': repository_config,
        'data_type': 'price',
        'align': True,
        'indicators': indicators or ['rsi', 'ma', 'macd', 'volatility']
    }
    
    # Create and execute pipeline
    pipeline = orchestrator.create_pipeline(context)
    return orchestrator.execute_pipeline(pipeline, context)
```

## Component Guide

### Data Loaders

The loaders are responsible for reading data from repositories:

- **PriceLoader**: Loads price data for currency pairs
- **MacroLoader**: Loads macroeconomic indicators
- **CustomLoader**: Flexible loader for custom data sources

```python
# Configure price loader
price_loader = PriceLoader(repository_path='/path/to/price/data')
orchestrator.register_component(price_loader)
```

### Validators

Validators check data quality and report issues:

- **DataValidator**: Validates price and macro data quality
- **GapDetector**: Detects and reports time gaps in data
- **QualityChecker**: Checks for data quality issues

```python
# Enable gap detection
request = {
    'detect_gaps': True,
    # other parameters...
}
```

### Processors

Processors transform data into a standardized format:

- **DataNormalizer**: Normalizes price and macro data format
- **FeatureCreator**: Creates trading features from data
- **DataTransformer**: Applies transformations to data

```python
# Normalization ensures consistent column names (open, high, low, close)
normalizer = DataNormalizer()
orchestrator.register_component(normalizer)
```

### Aligners

Aligners synchronize different time series:

- **TimeAligner**: Aligns price and macro data
- **TimeResampler**: Changes data timeframe (e.g., 1H → 4H)

```python
# Resample price data to a different timeframe
request = {
    'target_timeframe': '4H',
    'resample_method': 'last'
}
resampler = TimeResampler()
resampled_data = resampler.process(price_data, request)
```

### Indicators

Balance Breaker supports two indicator architectures:

#### Original Monolithic Indicators

Comprehensive classes that calculate multiple indicators:

- **TechnicalIndicators**: All technical indicators in one class
- **EconomicIndicators**: Economic indicators (yield spreads, inflation differentials)
- **CompositeIndicators**: Balance Breaker specific indicators (precession, market mood)

```python
# Calculate technical indicators
tech_indicators = TechnicalIndicators({
    'sma_periods': [10, 20, 50, 200],
    'rsi_period': 14
})
price_data_with_indicators = tech_indicators.process(price_data, {})
```

#### Modular Indicators

Individual, focused classes for specific indicators:

- **RSIIndicator**: Relative Strength Index
- **MovingAverageIndicator**: Simple and exponential moving averages
- **MACDIndicator**: Moving Average Convergence Divergence
- **MomentumIndicator**: Price momentum and rate of change
- **VolatilityIndicator**: Price volatility measures

```python
# Register all modular indicators
orchestrator.register_modular_indicators()

# Request specific indicators by name
request = {
    # other parameters...
    'indicators': ['rsi', 'ma', 'macd', 'momentum', 'volatility']
}
```

### Serializers

Serializers save and cache data:

- **DataExporter**: Exports data to various formats
- **CacheManager**: Caches pipeline results

```python
# Export data to CSV
exporter = DataExporter()
exporter.process(result, {
    'export_formats': ['csv'],
    'export_dir': 'exported_data',
    'export_prefix': 'backtest_data'
})
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
        
        # Check for Balance Breaker indicators
        if 'precession' in macro_df.columns:
            print("Last 5 precession values:")
            print(macro_df['precession'].tail(5))
```

## Choosing Between Indicator Architectures

Balance Breaker supports both monolithic and modular indicator systems. Here's when to use each:

### Use Original Monolithic Indicators When:

- You need complex interdependent calculations
- Performance optimization is critical 
- You need the full suite of indicators in one pass
- Your indicators require cross-indicator knowledge

### Use Modular Indicators When:

- You need specific, individual indicators
- You want clear separation of responsibilities
- You want to extend the system with custom indicators
- You need better testability and maintainability

## Advanced Usage

### Caching

The pipeline supports caching to improve performance:

```python
# Enable caching
orchestrator = DataPipelineOrchestrator({
    'cache_enabled': True,
    'cache_ttl': 3600  # 1 hour
})

# Register cache manager
cache_manager = CacheManager({
    'cache_dir': 'cache',
    'cache_ttl': 3600
})
orchestrator.register_component(cache_manager)
```

### Creating Custom Modular Indicators

You can create custom modular indicators:

```python
from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.data_pipeline.indicators.modular_base import ModularIndicator, register_indicator

@register_indicator
@implements("IndicatorCalculator")
class MyCustomIndicator(ModularIndicator):
    """My custom indicator implementation"""
    
    indicator_name = "mycustom"  # Used in requests
    indicator_category = "technical"
    required_columns = {"close"}
    
    def __init__(self, parameters=None):
        default_params = {
            'period': 14,
        }
        super().__init__(parameters or default_params)
    
    def calculate_indicator(self, df, **kwargs):
        period = self.parameters.get('period', 14)
        price = df['close']
        
        # Custom calculation logic
        result = price.rolling(window=period).mean() / price.rolling(window=period).std()
        
        return {'MyCustom': result}
```

### Custom Pipeline Components

You can create custom components by implementing the PipelineComponent interface:

```python
from balance_breaker.src.data_pipeline import PipelineComponent

class MyCustomProcessor(PipelineComponent):
    @property
    def component_type(self) -> str:
        return 'processor'
        
    def process(self, data, context):
        # Custom processing logic
        return processed_data
```

## Best Practices

1. **Register components in order**: Components are executed in registration order
2. **Use register_modular_indicators()**: Register all modular indicators with one call
3. **Request indicators by name**: Specify exactly which indicators you need
4. **Validate data**: Always include validators to detect issues early
5. **Use caching**: Enable caching for better performance
6. **Handle errors**: Wrap pipeline execution in try/except blocks
7. **Set appropriate context**: Provide all necessary information in the request context