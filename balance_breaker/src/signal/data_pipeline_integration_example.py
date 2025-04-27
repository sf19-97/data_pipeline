"""
Data Pipeline and Signal Subsystem Integration Example

This example demonstrates how the refactored Data Pipeline subsystem integrates with
the Signal Subsystem to create a complete workflow from data loading to signal generation.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from balance_breaker.src.core.integration_tools import event_bus

# Import Data Pipeline components
from balance_breaker.src.data_pipeline import (
    DataPipelineOrchestrator,
    InMemoryDataRegistry,
    PriceLoader,
    MacroLoader,
    DataValidator,
    DataNormalizer,
    TimeAligner
)
from balance_breaker.src.data_pipeline.indicators.technical_modular import (
    RSIIndicator,
    MovingAverageIndicator,
    MACDIndicator,
    VolatilityIndicator
)

# Import Signal Subsystem components
from balance_breaker.src.signal import (
    SignalOrchestrator,
    InMemorySignalRegistry,
    MovingAverageCrossSignalGenerator,
    PatternRecognitionSignalGenerator,
    ConfidenceFilter,
    TimeframeFilter,
    CorrelationSignalCombiner,
    TimeframeCombiner,
    SignalEnricherProcessor,
    Signal
)

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_data_pipeline() -> DataPipelineOrchestrator:
    """Set up the data pipeline for loading and processing market data"""
    logger.info("Setting up data pipeline...")
    
    # Create the data registry
    registry = InMemoryDataRegistry({
        'auto_clean_interval': 3600,  # 1 hour
        'enable_auto_clean': True,
        'max_data_age': 86400*7      # 7 days
    })
    
    # Create data pipeline orchestrator
    orchestrator = DataPipelineOrchestrator({
        'auto_register_data': True,
        'enable_event_processing': True
    })
    
    # Set the registry
    orchestrator.set_registry(registry)
    
    # Register components using type-specific methods
    orchestrator.register_loader(PriceLoader())
    orchestrator.register_loader(MacroLoader())
    orchestrator.register_validator(DataValidator())
    orchestrator.register_processor(DataNormalizer())
    orchestrator.register_aligner(TimeAligner())
    
    # Register individual indicators
    orchestrator.register_indicator(RSIIndicator({'period': 14}))
    orchestrator.register_indicator(MovingAverageIndicator({
        'periods': [8, 21, 50, 200],
        'types': ['simple', 'exponential']
    }))
    orchestrator.register_indicator(MACDIndicator({
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9
    }))
    orchestrator.register_indicator(VolatilityIndicator({
        'atr_period': 14,
        'calculate_bbands': True
    }))
    
    # Create standard pipeline configurations
    price_analysis_pipeline = {
        'id': 'price_analysis',
        'loaders': ['PriceLoader'],
        'validators': ['DataValidator'],
        'processors': ['DataNormalizer'],
        'aligners': ['TimeAligner'],
        'indicators': ['RSIIndicator', 'MovingAverageIndicator', 'MACDIndicator', 'VolatilityIndicator'],
        'auto_register_data': True
    }
    
    macro_analysis_pipeline = {
        'id': 'macro_analysis',
        'loaders': ['MacroLoader'],
        'validators': ['DataValidator'],
        'processors': ['DataNormalizer'],
        'auto_register_data': True
    }
    
    # Register pipeline configurations
    orchestrator.create_pipeline(price_analysis_pipeline)
    orchestrator.create_pipeline(macro_analysis_pipeline)
    
    logger.info("Data pipeline setup complete")
    return orchestrator


def setup_signal_subsystem() -> SignalOrchestrator:
    """Set up the signal subsystem for generating and processing signals"""
    logger.info("Setting up signal subsystem...")
    
    # Create signal registry
    registry = InMemorySignalRegistry({
        'auto_clean_interval': 300,  # 5 minutes
        'enable_auto_clean': True,
        'max_signal_age': 86400      # 24 hours
    })
    
    # Create signal orchestrator
    orchestrator = SignalOrchestrator({
        'auto_register_signals': True,
        'enable_event_processing': True
    })
    
    # Set registry
    orchestrator.set_registry(registry)
    
    # Register signal generators
    orchestrator.register_generator(MovingAverageCrossSignalGenerator({
        'fast_period': 8,
        'slow_period': 21,
        'signal_type': 'interpreted'
    }))
    
    orchestrator.register_generator(PatternRecognitionSignalGenerator({
        'patterns': ['doji', 'engulfing', 'hammer', 'shooting_star'],
        'signal_type': 'interpreted'
    }))
    
    # Register signal filters
    orchestrator.register_filter(ConfidenceFilter({
        'min_confidence': 2,
        'min_strength': 2
    }))
    
    orchestrator.register_filter(TimeframeFilter())
    
    # Register signal combiners
    orchestrator.register_combiner(CorrelationSignalCombiner())
    orchestrator.register_combiner(TimeframeCombiner())
    
    # Register signal processors
    orchestrator.register_processor(SignalEnricherProcessor())
    
    # Create signal pipelines
    technical_pipeline = {
        'id': 'technical_pipeline',
        'generators': ['MovingAverageCrossSignalGenerator'],
        'processors': ['SignalEnricherProcessor'],
        'filters': ['ConfidenceFilter'],
        'combiners': ['CorrelationSignalCombiner'],
        'data_types': ['price']
    }
    
    pattern_pipeline = {
        'id': 'pattern_pipeline',
        'generators': ['PatternRecognitionSignalGenerator'],
        'processors': ['SignalEnricherProcessor'],
        'filters': ['ConfidenceFilter', 'TimeframeFilter'],
        'data_types': ['price']
    }
    
    multi_timeframe_pipeline = {
        'id': 'multi_timeframe_pipeline',
        'combiners': ['TimeframeCombiner'],
        'context': {
            'min_timeframes': 2,
            'weight_higher_timeframes': True
        }
    }
    
    # Register pipelines
    orchestrator.create_pipeline(technical_pipeline)
    orchestrator.create_pipeline(pattern_pipeline)
    orchestrator.create_pipeline(multi_timeframe_pipeline)
    
    logger.info("Signal subsystem setup complete")
    return orchestrator


def load_market_data(
    dp_orchestrator: DataPipelineOrchestrator,
    pairs: List[str],
    start_date: str,
    end_date: str = None,
    timeframe: str = "1h"
) -> Dict[str, Any]:
    """
    Load market data using the data pipeline
    
    Args:
        dp_orchestrator: Data pipeline orchestrator
        pairs: List of currency pairs to load
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (optional)
        timeframe: Timeframe to load (default: "1h")
    
    Returns:
        Dictionary containing loaded data
    """
    logger.info(f"Loading market data for {pairs} from {start_date} to {end_date or 'now'}")
    
    # Create execution context
    context = {
        'pairs': pairs,
        'start_date': start_date,
        'end_date': end_date,
        'timeframe': timeframe,
        'use_cache': True,
        'cache_ttl': 3600  # 1 hour
    }
    
    # Execute the price analysis pipeline with context
    result = dp_orchestrator.execute_pipeline('price_analysis', context)
    
    # Log data summary
    if isinstance(result, dict) and 'price' in result:
        for pair, df in result['price'].items():
            logger.info(f"Loaded {len(df)} rows of {timeframe} data for {pair}")
    
    return result


def generate_signals(
    signal_orchestrator: SignalOrchestrator,
    dp_orchestrator: DataPipelineOrchestrator,
    pairs: List[str],
    start_date: str,
    end_date: str = None,
    timeframe: str = "1h"
) -> Dict[str, List[Signal]]:
    """
    Generate signals from market data
    
    Args:
        signal_orchestrator: Signal orchestrator
        dp_orchestrator: Data pipeline orchestrator
        pairs: List of currency pairs
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (optional)
        timeframe: Timeframe for signal generation
    
    Returns:
        Dictionary of signals by pipeline
    """
    logger.info(f"Generating signals for {pairs} at {timeframe} timeframe")
    
    # Use the dedicated integration method to get data optimized for signal generation
    market_data = dp_orchestrator.get_data_for_signals({
        'pairs': pairs,
        'start_date': start_date,
        'end_date': end_date,
        'timeframe': timeframe,
        'indicators': ['RSIIndicator', 'MovingAverageIndicator', 'MACDIndicator', 'VolatilityIndicator']
    })
    
    # Create signal context
    context = {
        'timeframe': timeframe,
        'timestamp': datetime.now(),
        # Add market context that enrichers can use
        'market_data': {
            pair: {
                'price': df.iloc[-1].to_dict(),
                'atr': df['ATR'].iloc[-1] if 'ATR' in df.columns else None,
                'volatility': 'high' if ('ATR' in df.columns and df['ATR'].iloc[-1] > df['ATR'].mean() * 1.5) 
                             else 'normal' if 'ATR' in df.columns else 'unknown'
            } for pair, df in market_data.get('price', {}).items()
        }
    }
    
    results = {}
    
    # Execute technical pipeline
    technical_signals = signal_orchestrator.execute_pipeline(
        'technical_pipeline', market_data, context
    )
    results['technical'] = technical_signals
    logger.info(f"Generated {len(technical_signals)} technical signals")
    
    # Execute pattern pipeline
    pattern_signals = signal_orchestrator.execute_pipeline(
        'pattern_pipeline', market_data, context
    )
    results['pattern'] = pattern_signals
    logger.info(f"Generated {len(pattern_signals)} pattern signals")
    
    # Combine all signals for multi-timeframe analysis
    all_signals = technical_signals + pattern_signals
    
    if len(all_signals) >= 2:
        # Execute multi-timeframe pipeline with all signals
        signals_data = {'signals': all_signals}
        multi_tf_signals = signal_orchestrator.execute_pipeline(
            'multi_timeframe_pipeline', signals_data, context
        )
        results['multi_timeframe'] = multi_tf_signals
        logger.info(f"Generated {len(multi_tf_signals)} multi-timeframe signals")
    
    return results


def analyze_signal_results(results: Dict[str, List[Signal]]) -> None:
    """
    Analyze signal generation results
    
    Args:
        results: Dictionary of signals by pipeline
    """
    total_count = sum(len(signals) for signals in results.values())
    logger.info(f"Analysis of {total_count} generated signals:")
    
    for pipeline_name, signals in results.items():
        if not signals:
            continue
            
        logger.info(f"Pipeline: {pipeline_name} ({len(signals)} signals)")
        
        # Count by symbol
        symbol_counts = {}
        for signal in signals:
            symbol = signal.symbol
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        logger.info(f"  Symbols: {symbol_counts}")
        
        # Count by direction
        direction_counts = {
            'BULLISH': len([s for s in signals if s.direction.name == 'BULLISH']),
            'BEARISH': len([s for s in signals if s.direction.name == 'BEARISH']),
            'NEUTRAL': len([s for s in signals if s.direction.name == 'NEUTRAL'])
        }
        
        logger.info(f"  Directions: {direction_counts}")
        
        # Show example signal
        if signals:
            example = signals[0]
            logger.info(f"  Example: {example.symbol} {example.direction.name} "
                      f"({example.strength.name} strength, {example.metadata.confidence.name} confidence)")
            
            if hasattr(example, 'interpretation') and example.interpretation:
                logger.info(f"  Interpretation: {example.interpretation}")


def event_based_integration_demo(
    dp_orchestrator: DataPipelineOrchestrator,
    signal_orchestrator: SignalOrchestrator
):
    """
    Demonstrate event-based integration between subsystems
    
    Args:
        dp_orchestrator: Data pipeline orchestrator
        signal_orchestrator: Signal orchestrator
    """
    logger.info("Setting up event-based integration demo")
    
    # Define event handler for data pipeline events
    def handle_pipeline_executed(event_data):
        pipeline_id = event_data.get('pipeline_id')
        logger.info(f"Received pipeline executed event for pipeline: {pipeline_id}")
        
        # Only process certain pipelines
        if pipeline_id != 'price_analysis':
            return
        
        # Get the data from the registry
        if not dp_orchestrator.registry:
            logger.warning("No data registry available")
            return
        
        # Find the latest price data in the registry
        results = dp_orchestrator.registry.find_data({
            'data_type': 'price',
            'tags': 'pipeline_result',
            'active_only': True
        })
        
        if not results:
            logger.warning("No price data found in registry")
            return
        
        # Use the most recent data
        latest_data = max(results, key=lambda x: x['metadata'].get('creation_time', datetime.min))
        
        # Get timeframe from metadata
        timeframe = latest_data['metadata'].get('timeframe', '1h')
        pairs = latest_data['metadata'].get('pairs', [])
        
        logger.info(f"Processing data for {pairs} at {timeframe} timeframe")
        
        # Process data through signal subsystem
        context = {
            'timeframe': timeframe,
            'data_source': 'event_based',
            'pairs': pairs
        }
        
        # Generate signals
        signals = signal_orchestrator.handle_data_update(latest_data['data'], context)
        
        logger.info(f"Generated {len(signals)} signals from data pipeline event")
        
        # Analyze the generated signals
        if signals:
            pipeline_results = {'event_based': signals}
            analyze_signal_results(pipeline_results)
    
    # Subscribe to pipeline executed events
    event_bus.subscribe('pipeline_executed', handle_pipeline_executed)
    
    # Simulate data being processed through pipeline
    pairs = ['EURUSD', 'USDJPY']
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    # Load data through pipeline
    load_market_data(dp_orchestrator, pairs, start_date, timeframe='1h')
    
    logger.info("Event-based integration demo complete")


def main():
    """Main function to demonstrate data pipeline integration"""
    logger.info("Starting data pipeline integration example")
    
    # Set up subsystems
    dp_orchestrator = setup_data_pipeline()
    signal_orchestrator = setup_signal_subsystem()
    
    # Define pairs and date range
    pairs = ['EURUSD', 'USDJPY', 'GBPUSD']
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Process multiple timeframes
    timeframes = ['1h', '4h', '1d']
    
    for timeframe in timeframes:
        logger.info(f"Processing {timeframe} timeframe data")
        
        # Generate signals using the integrated approach
        signals = generate_signals(
            signal_orchestrator, dp_orchestrator, 
            pairs, start_date, timeframe=timeframe
        )
        
        # Analyze signal results
        analyze_signal_results(signals)
    
    # Show event-based integration
    event_based_integration_demo(dp_orchestrator, signal_orchestrator)
    
    # Demonstrate registry capabilities
    if dp_orchestrator.registry:
        # Print registry statistics
        stored_data = dp_orchestrator.registry.find_data({'active_only': True})
        logger.info(f"Data registry contains {len(stored_data)} active data entries")
        
        # Group by data type
        data_types = {}
        for entry in stored_data:
            data_type = entry['metadata'].get('data_type', 'unknown')
            data_types[data_type] = data_types.get(data_type, 0) + 1
        
        logger.info(f"Data types in registry: {data_types}")
        
        # Clean expired data
        cleaned = dp_orchestrator.registry.clean_expired_data()
        logger.info(f"Cleaned {cleaned} expired data entries from registry")
    
    logger.info("Data pipeline integration example complete")


if __name__ == "__main__":
    main()