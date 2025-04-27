"""
Balance Breaker Signal Subsystem

This module provides components for signal generation, processing, and management.
The signal subsystem serves as the nervous system of the trading platform,
detecting market conditions and coordinating actions across other subsystems.
"""

# Export key models for easier imports
from balance_breaker.src.signal.signal_models import (
    Signal, RawSignal, InterpretedSignal, ActionSignal,
    SignalGroup, SignalMetadata, SignalType, SignalDirection,
    SignalStrength, SignalPriority, SignalConfidence, Timeframe
)

# Export core interface classes
from balance_breaker.src.signal.signal_interfaces import (
    SignalComponent, SignalGenerator, SignalProcessor,
    SignalFilter, SignalCombiner, SignalRegistry, SignalConsumer
)

# Export base classes
from balance_breaker.src.signal.base import (
    BaseSignalComponent, BaseSignalGenerator, BaseSignalProcessor,
    BaseSignalFilter, BaseSignalCombiner
)

# Export signal registry
from balance_breaker.src.signal.signal_registry import (
    InMemorySignalRegistry, SignalRegistryError
)

# Export signal orchestrator
from balance_breaker.src.signal.signal_orchestrator import (
    SignalOrchestrator, SignalOrchestratorError
)

# Export signal generators
from balance_breaker.src.signal.generators.ma_cross_generator import MovingAverageCrossSignalGenerator
from balance_breaker.src.signal.generators.pattern_generator import PatternRecognitionSignalGenerator

# Export signal filters
from balance_breaker.src.signal.filters.confidence_filter import ConfidenceFilter
from balance_breaker.src.signal.filters.timeframe_filter import TimeframeFilter
from balance_breaker.src.signal.filters.direction_filter import DirectionFilter

# Export signal combiners
from balance_breaker.src.signal.combiners.correlation_combiner import CorrelationSignalCombiner
from balance_breaker.src.signal.combiners.timeframe_combiner import TimeframeCombiner

# Export signal processors
from balance_breaker.src.signal.processors.signal_enricher_processor import SignalEnricherProcessor

# Version information
__version__ = '0.1.0'