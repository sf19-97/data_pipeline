# Balance Breaker Signal Subsystem

The Signal Subsystem serves as the nervous system of the Balance Breaker trading platform. It processes market data to detect conditions, generate trading signals, and coordinate actions across the platform.

## Architecture Overview

The Signal Subsystem follows a modular architecture with these key components:

1. **Signal Models**: Core data structures representing different types of signals with metadata
2. **Signal Registry**: Central storage for managing signal lifecycle and relationships
3. **Signal Orchestrator**: Coordinates signal generation, processing, filtering, and combining
4. **Signal Generators**: Convert market data into raw, interpreted, or action signals
5. **Signal Processors**: Transform signals to add context or enhance information
6. **Signal Filters**: Select signals based on specific criteria
7. **Signal Combiners**: Merge related signals to create higher-quality composite signals

## Signal Hierarchy

Signals follow a three-layer hierarchy:

1. **Raw Signals**: Direct observations without implied action (e.g., "moving average crossover detected")
2. **Interpreted Signals**: Market meaning without specific action (e.g., "bullish trend reversal likely")
3. **Action Signals**: Specific actions for execution (e.g., "buy EURUSD at 1.1250")

## Data Pipeline Integration

The Signal Subsystem integrates with the Data Pipeline subsystem to convert processed market data into actionable signals:

```
Data Pipeline                     Signal Subsystem
+---------------+                +------------------+
| Price Data    |                | Signal           |
| Technical     |----Data------->| Generators       |
| Indicators    |                |                  |
| Economic Data |                | Signal           |
+---------------+                | Orchestrator     |
                                 +------------------+
                                          |
                                          v
                                 +------------------+
                                 | Signal           |
                                 | Processors       |
                                 |                  |
                                 | Signal           |
                                 | Filters          |
                                 +------------------+
                                          |
                                          v
                                 +------------------+
                                 | Signal           |
                                 | Combiners        |
                                 |                  |
                                 | Signal           |
                                 | Registry         |
                                 +------------------+
                                          |
                                          v
                                 +------------------+
                                 | To Strategy &    |
                                 | Portfolio        |
                                 | Subsystems       |
                                 +------------------+
```

## Getting Started

### Basic Usage

```python
from balance_breaker.src.signal import (
    SignalOrchestrator,
    InMemorySignalRegistry,
    MovingAverageCrossSignalGenerator,
    ConfidenceFilter,
    CorrelationSignalCombiner
)

# Create and configure registry
registry = InMemorySignalRegistry()

# Create and configure orchestrator
orchestrator = SignalOrchestrator()
orchestrator.set_registry(registry)

# Register components
orchestrator.register_generator(MovingAverageCrossSignalGenerator())
orchestrator.register_filter(ConfidenceFilter())
orchestrator.register_combiner(CorrelationSignalCombiner())

# Create a pipeline configuration
pipeline_config = {
    'id': 'ma_cross_pipeline',
    'generators': ['MovingAverageCrossSignalGenerator'],
    'filters': ['ConfidenceFilter'],
    'combiners': ['CorrelationSignalCombiner'],
    'auto_register': True,
    'data_types': ['price']
}

# Create the pipeline
pipeline_id = orchestrator.create_pipeline(pipeline_config)

# Generate signals from data
price_data = {
    'EURUSD': df_eurusd,
    'GBPUSD': df_gbpusd
}

context = {
    'timeframe': '1h',
    'min_confidence': 2,
    'min_strength': 2
}

signals = orchestrator.execute_pipeline(pipeline_id, price_data, context)
```

### Data Pipeline Integration

```python
from balance_breaker.src.data_pipeline import (
    DataPipelineOrchestrator,
    PriceLoader,
    MacroLoader,
    DataValidator,
    DataNormalizer,
    TimeAligner,
    TechnicalIndicators
)

from balance_breaker.src.signal import (
    SignalOrchestrator,
    InMemorySignalRegistry
)

# Set up Data Pipeline
dp_orchestrator = DataPipelineOrchestrator()
dp_orchestrator.register_component(PriceLoader())
dp_orchestrator.register_component(TechnicalIndicators())
# Register other components...

# Set up Signal Subsystem
signal_registry = InMemorySignalRegistry()
signal_orchestrator = SignalOrchestrator()
signal_orchestrator.set_registry(signal_registry)
# Register signal components...

# Load data using the Data Pipeline
data_request = {
    'pairs': ['EURUSD', 'USDJPY'],
    'start_date': '2023-01-01',
    'end_date': '2023-02-01',
    'data_type': 'price',
    'indicators': ['TechnicalIndicators']
}
data_pipeline = dp_orchestrator.create_pipeline(data_request)
data_result = dp_orchestrator.execute_pipeline(data_pipeline, data_request)

# Process data through Signal Subsystem
signal_context = {
    'timeframe': '1h',
    'data_source': 'data_pipeline'
}
signals = signal_orchestrator.handle_data_update(data_result, signal_context)
```

## Component Guide

### Signal Registry

The registry manages the lifecycle of signals:

```python
# Query signals by various criteria
active_signals = registry.find_signals({
    'symbol': 'EURUSD',
    'timeframe': '1h',
    'active_only': True
})

# Get a specific signal
signal = registry.get_signal(signal_id)

# Create parent-child relationships
registry.link_signals(parent_id, child_id)

# Clean expired signals
registry.clean_expired_signals()
```

### Signal Generators

Available signal generators:

1. **MovingAverageCrossSignalGenerator**: Generates signals based on moving average crossovers
2. **PatternRecognitionSignalGenerator**: Generates signals based on candlestick patterns

### Signal Filters

Available signal filters:

1. **ConfidenceFilter**: Filters signals based on confidence/strength thresholds 
2. **TimeframeFilter**: Filters signals by timeframe
3. **DirectionFilter**: Filters signals by direction

### Signal Combiners

Available signal combiners:

1. **CorrelationSignalCombiner**: Combines correlated signals
2. **TimeframeCombiner**: Combines signals across timeframes

### Signal Processors

Available signal processors:

1. **SignalEnricherProcessor**: Enriches signals with additional market context

## Timeframe Handling

The Signal Subsystem implements a hierarchical timeframe model:

1. Higher timeframe signals cascade down with decreasing strength
2. Lower timeframe signals can confirm higher timeframe signals
3. Conflicting signals are resolved based on timeframe priority

## Best Practices

1. **Use hierarchical signal types**: Start with raw signals, process to interpreted signals, and finally create action signals
2. **Track signal relationships**: Link related signals using parent-child relationships to trace signal provenance
3. **Filter aggressively**: Use filters to reduce noise and focus on high-quality signals
4. **Validate signal quality**: Prioritize signals with higher confidence and strength
5. **Consider timeframe context**: Use higher timeframes for direction and lower timeframes for timing
6. **Set appropriate expiration**: Signals should expire when no longer relevant to prevent stale signals