"""
Data Pipeline Integration Example

This example demonstrates how the Signal Subsystem integrates with the Data Pipeline
to create a complete workflow from data loading to signal generation.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from balance_breaker.src.core.integration_tools import event_bus

# Import Data Pipeline components
from balance_breaker.src.data_pipeline import (
    DataPipelineOrchestrator,
    PriceLoader,
    MacroLoader,
    DataValidator,
    DataNormalizer,
    TimeAligner,
    TechnicalIndicators
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
    
    # Create data pipeline orchestrator
    orchestrator = DataPipelineOrchestrator()
    
    # Register components
    orchestrator.register_component(PriceLoader())
    orchestrator.register_component(DataValidator())
    orchestrator.register_component(DataNormalizer())
    orchestrator.register_component(TechnicalIndicators({
        'sma_periods': [8, 21, 50, 200],  # Match our signal generator parameters
        'rsi_period': 14,
        'macd_params': (12, 26, 9),
        'bbands_params': (20, 2),
        'generate_all': True
    }))
    
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
    
    # Create data request
    request = {
        'pairs': pairs,
        'start_date': start_date,
        'end_date': end_date,
        'data_type': 'price',
        'timeframe': timeframe,
        'indicators': ['TechnicalIndicators']
    }
    
    # Create and execute pipeline
    pipeline = dp_orchestrator.create_pipeline(request)
    result = dp_orchestrator.execute_pipeline(pipeline, request)
    
    # Log data summary
    if isinstance(result, dict) and 'price' in result:
        for pair, df in result['price'].items():
            logger.info(f"Loaded {len(df)} rows of {timeframe} data for {pair}")
    
    return result


def generate_signals(
    signal_orchestrator: SignalOrchestrator,
    market_data: Dict[str, Any],
    timeframe: str = "1h"
) -> Dict[str, List[Signal]]:
    """
    Generate signals from market data
    
    Args:
        signal_orchestrator: Signal orchestrator
        market_data: Market data from data pipeline
        timeframe: Timeframe for signal generation
    
    Returns:
        Dictionary of signals by pipeline
    """
    logger.info(f"Generating signals for {timeframe} timeframe")
    
    results = {}
    
    # Create signal context
    context = {
        'timeframe': timeframe,
        'timestamp': datetime.now(),
        # Add market context that enrichers can use
        'market_data': {
            pair: {
                'price': df.iloc[-1].to_dict(),
                'atr': df['ATR'].iloc[-1] if 'ATR' in df.columns else None,
                'volatility': 'high' if df['ATR'].iloc[-1] > df['ATR'].mean() * 1.5 else 'normal'
                if 'ATR' in df.columns else 'unknown'
            } for pair, df in market_data.get('price', {}).items()
        }
    }
    
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
            
            if hasattr(example, 'interpretation'):
                logger.info(f"  Interpretation: {example.interpretation}")


def event_based_integration_demo():
    """
    Demonstrate event-based integration between subsystems
    """
    logger.info("Setting up event-based integration demo")
    
    # Set up subsystems
    dp_orchestrator = setup_data_pipeline()
    signal_orchestrator = setup_signal_subsystem()
    
    # Define event handler for data pipeline updates
    def handle_data_pipeline_update(event_data):
        logger.info("Received data pipeline update event")
        
        data = event_data.get('data')
        context = event_data.get('context', {})
        
        if data:
            # Add timeframe to context if not present
            if 'timeframe' not in context:
                context['timeframe'] = '1h'  # Default timeframe
            
            # Process data through signal subsystem
            signals = signal_orchestrator.handle_data_update(data, context)
            
            logger.info(f"Generated {len(signals)} signals from data pipeline event")
    
    # Subscribe to data pipeline update events
    event_bus.subscribe('data_pipeline_update', handle_data_pipeline_update)
    
    # Simulate a data pipeline update event
    pairs = ['EURUSD', 'USDJPY']
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Load data
    data = load_market_data(dp_orchestrator, pairs, start_date)
    
    # Publish a data pipeline update event
    event_bus.publish('data_pipeline_update', {
        'data': data,
        'context': {
            'timeframe': '1h',
            'data_source': 'data_pipeline',
            'pairs': pairs
        }
    })
    
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
        # Load market data using data pipeline
        market_data = load_market_data(
            dp_orchestrator, pairs, start_date, timeframe=timeframe
        )
        
        # Generate signals using signal subsystem
        signals = generate_signals(
            signal_orchestrator, market_data, timeframe=timeframe
        )
        
        # Analyze signal results
        analyze_signal_results(signals)
    
    # Show event-based integration
    event_based_integration_demo()
    
    logger.info("Data pipeline integration example complete")


if __name__ == "__main__":
    main()