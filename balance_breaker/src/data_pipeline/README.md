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

# Define request
request = {
    'pairs': ['USDJPY'],
    'start_date': '2022-01-01',
    'end_date': '2023-01-01',
    'data_type': 'price',
    'align': True
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
    repository_config=None
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
        'indicators': ['EconomicIndicators', 'TechnicalIndicators', 'CompositeIndicators']
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

```python
# Configure price loader
price_loader = PriceLoader(repository_path='/path/to/price/data')
orchestrator.register_component(price_loader)
```

### Validators

Validators check data quality and report issues:

- **DataValidator**: Validates price and macro data quality

```python
# Validation results are added to the context
if 'validation' in context and context['validation']['status'] == 'warning':
    print("Data quality warnings:", context['validation']['issues'])
```

### Processors

Processors transform data into a standardized format:

- **DataNormalizer**: Normalizes price and macro data format

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

Indicators calculate derived metrics:

- **EconomicIndicators**: Economic indicators (yield spreads, inflation differentials)
- **TechnicalIndicators**: Technical indicators (moving averages, RSI, MACD)
- **CompositeIndicators**: Balance Breaker specific indicators (precession, market mood)

```python
# Calculate technical indicators
tech_indicators = TechnicalIndicators({
    'sma_periods': [10, 20, 50, 200],
    'rsi_period': 14
})
price_data_with_indicators = tech_indicators.process(price_data, {})
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
    
    # Access aligned macro data
    if pair in result['aligned_macro']:
        macro_df = result['aligned_macro'][pair]
        print(f"Macro data for {pair}: {len(macro_df)} rows")
        
        # Check for Balance Breaker indicators
        if 'precession' in macro_df.columns:
            print("Last 5 precession values:")
            print(macro_df['precession'].tail(5))
```

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

### Custom Components

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
2. **Validate data**: Always include validators to detect issues early
3. **Use caching**: Enable caching for better performance
4. **Handle errors**: Wrap pipeline execution in try/except blocks
5. **Set appropriate context**: Provide all necessary information in the request context