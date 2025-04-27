"""
Signal Orchestrator

This module provides the orchestrator for managing the flow of signals
through generators, processors, filters, and combiners.
"""

import logging
import threading
from typing import Dict, List, Any, Optional, Set, Union, Callable, Tuple
from datetime import datetime, timedelta
import json
import os
import copy

from balance_breaker.src.core.interface_registry import implements, registry
from balance_breaker.src.core.error_handling import ErrorHandler, BalanceBreakerError, ErrorSeverity, ErrorCategory
from balance_breaker.src.core.parameter_manager import ParameterizedComponent
from balance_breaker.src.core.integration_tools import event_bus, integrates_with, IntegrationType
from balance_breaker.src.signal.signal_interfaces import (
    SignalGenerator, SignalProcessor, SignalFilter, 
    SignalCombiner, SignalRegistry
)
from balance_breaker.src.signal.signal_models import (
    Signal, SignalGroup, SignalMetadata, Timeframe,
    SignalType, SignalDirection, SignalStrength
)


class SignalOrchestratorError(BalanceBreakerError):
    """Error in signal orchestrator operations"""
    
    def __init__(self, 
                message: str, 
                component: str = "",
                severity: ErrorSeverity = ErrorSeverity.ERROR,
                context: Optional[Dict[str, Any]] = None,
                original_exception: Optional[Exception] = None):
        super().__init__(
            message=message, 
            subsystem="signal",
            component=component,
            severity=severity,
            category=ErrorCategory.EXECUTION,
            context=context,
            original_exception=original_exception
        )


class SignalOrchestrator(ParameterizedComponent):
    """
    Central orchestrator for the signal subsystem
    
    This orchestrator manages the flow of signals through generators,
    processors, filters, and combiners. It supports both synchronous
    processing and event-based signal handling.
    
    Parameters:
    -----------
    auto_register_signals : bool
        Whether to automatically register generated signals (default: True)
    enable_event_processing : bool
        Whether to enable event-based signal processing (default: True)
    timeframe_priority : List[str]
        Priority order for timeframes (default: ['1M', '1w', '1d', '4h', '1h', '30m', '15m', '5m', '1m'])
    registry_id : str
        ID of the signal registry to use (default: 'default')
    pipeline_config_path : str
        Path to pipeline configuration file (default: None)
    """
    
    def __init__(self, parameters=None):
        """Initialize signal orchestrator with optional parameters"""
        # Define default parameters
        default_params = {
            'auto_register_signals': True,
            'enable_event_processing': True,
            'timeframe_priority': ['1M', '1w', '1d', '4h', '1h', '30m', '15m', '5m', '1m'],
            'registry_id': 'default',
            'pipeline_config_path': None
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
        
        # Set up logging and error handling
        self.logger = logging.getLogger(__name__)
        self.error_handler = ErrorHandler(self.logger)
        
        # Component registries
        self.generators: Dict[str, SignalGenerator] = {}
        self.processors: Dict[str, SignalProcessor] = {}
        self.filters: Dict[str, SignalFilter] = {}
        self.combiners: Dict[str, SignalCombiner] = {}
        
        # Signal registry reference
        self.registry_id = self.parameters.get('registry_id', 'default')
        self.registry = None
        
        # Pipeline configurations
        self.pipelines: Dict[str, Dict[str, Any]] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Thread lock
        self._lock = threading.RLock()
        
        # Load pipeline configurations if provided
        pipeline_config_path = self.parameters.get('pipeline_config_path')
        if pipeline_config_path and os.path.exists(pipeline_config_path):
            self._load_pipeline_config(pipeline_config_path)
        
        # Register event handlers if enabled
        if self.parameters.get('enable_event_processing', True):
            self._register_event_handlers()
    
    def set_registry(self, registry: SignalRegistry) -> None:
        """
        Set the signal registry
        
        Args:
            registry: Signal registry to use
        """
        self.registry = registry
    
    def register_generator(self, generator: SignalGenerator) -> bool:
        """
        Register a signal generator
        
        Args:
            generator: Signal generator to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            # Validate generator implements SignalGenerator interface
            validation = registry.validate_implementation(generator, "SignalGenerator")
            if not validation.get('valid', False):
                self.logger.error(f"Invalid signal generator: {validation}")
                return False
            
            with self._lock:
                name = generator.name
                self.generators[name] = generator
                self.logger.info(f"Registered signal generator: {name}")
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'generator_name': getattr(generator, 'name', None)},
                subsystem='signal',
                component='SignalOrchestrator'
            )
            return False
    
    def register_processor(self, processor: SignalProcessor) -> bool:
        """
        Register a signal processor
        
        Args:
            processor: Signal processor to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            # Validate processor implements SignalProcessor interface
            validation = registry.validate_implementation(processor, "SignalProcessor")
            if not validation.get('valid', False):
                self.logger.error(f"Invalid signal processor: {validation}")
                return False
            
            with self._lock:
                name = processor.name
                self.processors[name] = processor
                self.logger.info(f"Registered signal processor: {name}")
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'processor_name': getattr(processor, 'name', None)},
                subsystem='signal',
                component='SignalOrchestrator'
            )
            return False
    
    def register_filter(self, filter_component: SignalFilter) -> bool:
        """
        Register a signal filter
        
        Args:
            filter_component: Signal filter to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            # Validate filter implements SignalFilter interface
            validation = registry.validate_implementation(filter_component, "SignalFilter")
            if not validation.get('valid', False):
                self.logger.error(f"Invalid signal filter: {validation}")
                return False
            
            with self._lock:
                name = filter_component.name
                self.filters[name] = filter_component
                self.logger.info(f"Registered signal filter: {name}")
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'filter_name': getattr(filter_component, 'name', None)},
                subsystem='signal',
                component='SignalOrchestrator'
            )
            return False
    
    def register_combiner(self, combiner: SignalCombiner) -> bool:
        """
        Register a signal combiner
        
        Args:
            combiner: Signal combiner to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            # Validate combiner implements SignalCombiner interface
            validation = registry.validate_implementation(combiner, "SignalCombiner")
            if not validation.get('valid', False):
                self.logger.error(f"Invalid signal combiner: {validation}")
                return False
            
            with self._lock:
                name = combiner.name
                self.combiners[name] = combiner
                self.logger.info(f"Registered signal combiner: {name}")
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'combiner_name': getattr(combiner, 'name', None)},
                subsystem='signal',
                component='SignalOrchestrator'
            )
            return False
    
    def create_pipeline(self, config: Dict[str, Any]) -> str:
        """
        Create a signal processing pipeline
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Pipeline ID
        """
        try:
            # Generate pipeline ID
            pipeline_id = config.get('id', f"pipeline_{len(self.pipelines) + 1}")
            
            with self._lock:
                # Store pipeline configuration
                self.pipelines[pipeline_id] = config
                
                self.logger.info(f"Created signal pipeline: {pipeline_id}")
                return pipeline_id
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'config': config},
                subsystem='signal',
                component='SignalOrchestrator'
            )
            raise SignalOrchestratorError(
                message=f"Failed to create pipeline: {str(e)}",
                component='SignalOrchestrator',
                context={'config': config},
                original_exception=e
            )
    
    @integrates_with(
        target_subsystem='data_pipeline',
        integration_type=IntegrationType.DATA_FLOW,
        description='Generates signals from data pipeline output'
    )
    def generate_signals_from_data(self, data: Any, context: Dict[str, Any]) -> List[Signal]:
        """
        Generate signals from data pipeline output
        
        Args:
            data: Data from data pipeline
            context: Generation context
            
        Returns:
            List of generated signals
        """
        try:
            all_signals = []
            
            # Determine which generators to use
            generator_names = context.get('generators', list(self.generators.keys()))
            
            # Generate signals from each generator
            for name in generator_names:
                if name not in self.generators:
                    self.logger.warning(f"Generator not found: {name}")
                    continue
                
                generator = self.generators[name]
                
                # Check if generator can generate signals from this data
                if not generator.can_generate(data, context):
                    self.logger.debug(f"Generator {name} cannot generate signals from provided data")
                    continue
                
                # Generate signals
                try:
                    signals = generator.generate_signals(data, context)
                    self.logger.info(f"Generator {name} produced {len(signals)} signals")
                    
                    # Process generated signals
                    processed_signals = self._process_generated_signals(signals, context)
                    all_signals.extend(processed_signals)
                    
                except Exception as e:
                    self.error_handler.handle_error(
                        e,
                        context={'generator': name},
                        subsystem='signal',
                        component='SignalOrchestrator'
                    )
            
            # Register signals if auto-register is enabled
            if self.parameters.get('auto_register_signals', True) and self.registry:
                for signal in all_signals:
                    self.registry.register_signal(signal)
            
            # Publish signals generated event
            event_bus.publish('signals_generated', {
                'count': len(all_signals),
                'context': {k: v for k, v in context.items() if isinstance(v, (str, int, float, bool))}
            })
            
            return all_signals
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'operation': 'generate_signals_from_data'},
                subsystem='signal',
                component='SignalOrchestrator'
            )
            return []
    
    def process_signals(self, signals: List[Signal], context: Dict[str, Any]) -> List[Signal]:
        """
        Process signals through processors
        
        Args:
            signals: Signals to process
            context: Processing context
            
        Returns:
            Processed signals
        """
        try:
            if not signals:
                return []
            
            # Make a copy to avoid modifying original signals
            working_signals = copy.deepcopy(signals)
            
            # Determine which processors to use
            processor_names = context.get('processors', list(self.processors.keys()))
            
            # Process signals through each processor
            for name in processor_names:
                if name not in self.processors:
                    self.logger.warning(f"Processor not found: {name}")
                    continue
                
                processor = self.processors[name]
                
                # Filter signals that this processor can handle
                processable_signals = [s for s in working_signals if processor.can_process(s)]
                
                if not processable_signals:
                    self.logger.debug(f"No signals for processor {name} to process")
                    continue
                
                # Process signals
                try:
                    processed = processor.process_signals(processable_signals, context)
                    
                    # Replace processed signals in the working set
                    processed_ids = {s.id for s in processed}
                    working_signals = [s for s in working_signals if s.id not in processed_ids] + processed
                    
                    self.logger.debug(f"Processor {name} processed {len(processable_signals)} signals")
                    
                except Exception as e:
                    self.error_handler.handle_error(
                        e,
                        context={'processor': name},
                        subsystem='signal',
                        component='SignalOrchestrator'
                    )
            
            return working_signals
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'operation': 'process_signals'},
                subsystem='signal',
                component='SignalOrchestrator'
            )
            return signals  # Return original signals on error
    
    def filter_signals(self, signals: List[Signal], context: Dict[str, Any]) -> List[Signal]:
        """
        Filter signals through filters
        
        Args:
            signals: Signals to filter
            context: Filtering context
            
        Returns:
            Filtered signals
        """
        try:
            if not signals:
                return []
            
            # Make a copy to avoid modifying original signals
            working_signals = copy.deepcopy(signals)
            
            # Determine which filters to use
            filter_names = context.get('filters', list(self.filters.keys()))
            
            # Filter signals through each filter
            for name in filter_names:
                if name not in self.filters:
                    self.logger.warning(f"Filter not found: {name}")
                    continue
                
                filter_component = self.filters[name]
                
                # Apply filter
                try:
                    filtered = filter_component.filter_signals(working_signals, context)
                    working_signals = filtered
                    
                    self.logger.debug(f"Filter {name} reduced signals from {len(signals)} to {len(filtered)}")
                    
                except Exception as e:
                    self.error_handler.handle_error(
                        e,
                        context={'filter': name},
                        subsystem='signal',
                        component='SignalOrchestrator'
                    )
            
            return working_signals
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'operation': 'filter_signals'},
                subsystem='signal',
                component='SignalOrchestrator'
            )
            return signals  # Return original signals on error
    
    def combine_signals(self, signals: List[Signal], context: Dict[str, Any]) -> List[Signal]:
        """
        Combine signals through combiners
        
        Args:
            signals: Signals to combine
            context: Combination context
            
        Returns:
            Original signals plus combined signals
        """
        try:
            if not signals:
                return []
            
            # Make a copy to avoid modifying original signals
            result_signals = copy.deepcopy(signals)
            
            # Determine which combiners to use
            combiner_names = context.get('combiners', list(self.combiners.keys()))
            
            # Combine signals through each combiner
            for name in combiner_names:
                if name not in self.combiners:
                    self.logger.warning(f"Combiner not found: {name}")
                    continue
                
                combiner = self.combiners[name]
                
                # Check if combiner can work with these signals
                if not combiner.can_combine(result_signals):
                    self.logger.debug(f"Combiner {name} cannot combine the provided signals")
                    continue
                
                # Combine signals
                try:
                    combined = combiner.combine_signals(result_signals, context)
                    
                    # Add combined signals to the result
                    result_signals.extend(combined)
                    
                    self.logger.debug(f"Combiner {name} produced {len(combined)} new signals")
                    
                except Exception as e:
                    self.error_handler.handle_error(
                        e,
                        context={'combiner': name},
                        subsystem='signal',
                        component='SignalOrchestrator'
                    )
            
            return result_signals
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'operation': 'combine_signals'},
                subsystem='signal',
                component='SignalOrchestrator'
            )
            return signals  # Return original signals on error
    
    def execute_pipeline(self, pipeline_id: str, data: Any, context: Dict[str, Any]) -> List[Signal]:
        """
        Execute a signal processing pipeline
        
        Args:
            pipeline_id: ID of the pipeline to execute
            data: Input data for the pipeline
            context: Pipeline execution context
            
        Returns:
            List of signals produced by the pipeline
        """
        try:
            # Check if pipeline exists
            if pipeline_id not in self.pipelines:
                raise SignalOrchestratorError(
                    message=f"Pipeline not found: {pipeline_id}",
                    component='SignalOrchestrator',
                    context={'pipeline_id': pipeline_id}
                )
            
            # Get pipeline configuration
            pipeline_config = self.pipelines[pipeline_id]
            
            # Create merged context
            merged_context = {**context}
            if 'context' in pipeline_config:
                merged_context.update(pipeline_config['context'])
            
            # Track execution metrics
            start_time = datetime.now()
            metrics = {
                'pipeline_id': pipeline_id,
                'start_time': start_time,
                'stages': {}
            }
            
            # Execute pipeline stages
            result_signals = []
            
            # 1. Generate signals
            if 'generators' in pipeline_config:
                merged_context['generators'] = pipeline_config['generators']
                
                stage_start = datetime.now()
                signals = self.generate_signals_from_data(data, merged_context)
                stage_end = datetime.now()
                
                metrics['stages']['generate'] = {
                    'duration': (stage_end - stage_start).total_seconds(),
                    'signal_count': len(signals)
                }
                
                result_signals = signals
            
            # 2. Process signals
            if 'processors' in pipeline_config and result_signals:
                merged_context['processors'] = pipeline_config['processors']
                
                stage_start = datetime.now()
                signals = self.process_signals(result_signals, merged_context)
                stage_end = datetime.now()
                
                metrics['stages']['process'] = {
                    'duration': (stage_end - stage_start).total_seconds(),
                    'signal_count': len(signals)
                }
                
                result_signals = signals
            
            # 3. Filter signals
            if 'filters' in pipeline_config and result_signals:
                merged_context['filters'] = pipeline_config['filters']
                
                stage_start = datetime.now()
                signals = self.filter_signals(result_signals, merged_context)
                stage_end = datetime.now()
                
                metrics['stages']['filter'] = {
                    'duration': (stage_end - stage_start).total_seconds(),
                    'signal_count': len(signals)
                }
                
                result_signals = signals
            
            # 4. Combine signals
            if 'combiners' in pipeline_config and result_signals:
                merged_context['combiners'] = pipeline_config['combiners']
                
                stage_start = datetime.now()
                signals = self.combine_signals(result_signals, merged_context)
                stage_end = datetime.now()
                
                metrics['stages']['combine'] = {
                    'duration': (stage_end - stage_start).total_seconds(),
                    'signal_count': len(signals)
                }
                
                result_signals = signals
            
            # Calculate overall metrics
            end_time = datetime.now()
            metrics['end_time'] = end_time
            metrics['duration'] = (end_time - start_time).total_seconds()
            metrics['total_signal_count'] = len(result_signals)
            
            # Log pipeline execution
            self.logger.info(
                f"Pipeline {pipeline_id} executed in {metrics['duration']:.3f}s, "
                f"produced {metrics['total_signal_count']} signals"
            )
            
            # Register results if auto-register is enabled
            if (pipeline_config.get('auto_register', self.parameters.get('auto_register_signals', True)) 
                and self.registry and result_signals):
                for signal in result_signals:
                    self.registry.register_signal(signal)
            
            # Publish pipeline execution event
            event_bus.publish('pipeline_executed', {
                'pipeline_id': pipeline_id,
                'signal_count': len(result_signals),
                'duration': metrics['duration']
            })
            
            return result_signals
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'pipeline_id': pipeline_id},
                subsystem='signal',
                component='SignalOrchestrator'
            )
            raise SignalOrchestratorError(
                message=f"Failed to execute pipeline {pipeline_id}: {str(e)}",
                component='SignalOrchestrator',
                context={'pipeline_id': pipeline_id},
                original_exception=e
            )
    
    def handle_data_update(self, data: Any, context: Dict[str, Any]) -> List[Signal]:
        """
        Handle data update event
        
        This method is called when new data is available from the data pipeline.
        It executes all pipelines that match the data type.
        
        Args:
            data: Updated data
            context: Update context
            
        Returns:
            List of all signals generated
        """
        try:
            all_signals = []
            data_type = context.get('data_type', 'unknown')
            
            # Find pipelines that match the data type
            matching_pipelines = []
            for pipeline_id, pipeline_config in self.pipelines.items():
                if pipeline_config.get('data_types', []):
                    if data_type in pipeline_config['data_types']:
                        matching_pipelines.append(pipeline_id)
                elif pipeline_config.get('auto_execute', False):
                    matching_pipelines.append(pipeline_id)
            
            # Execute matching pipelines
            for pipeline_id in matching_pipelines:
                try:
                    signals = self.execute_pipeline(pipeline_id, data, context)
                    all_signals.extend(signals)
                except Exception as e:
                    self.error_handler.handle_error(
                        e,
                        context={'pipeline_id': pipeline_id, 'data_type': data_type},
                        subsystem='signal',
                        component='SignalOrchestrator'
                    )
            
            return all_signals
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'data_type': context.get('data_type', 'unknown')},
                subsystem='signal',
                component='SignalOrchestrator'
            )
            return []
    
    def get_signals_for_timeframe(self, timeframe: Union[Timeframe, str], 
                                context: Dict[str, Any]) -> List[Signal]:
        """
        Get signals for a specific timeframe
        
        Args:
            timeframe: Target timeframe
            context: Query context
            
        Returns:
            List of signals for the timeframe
        """
        try:
            if not self.registry:
                return []
            
            # Convert timeframe to string if needed
            timeframe_value = timeframe.value if hasattr(timeframe, 'value') else timeframe
            
            # Build search criteria
            criteria = {
                'timeframe': timeframe_value,
                'active_only': context.get('active_only', True),
                **{k: v for k, v in context.items() if k not in ['active_only']}
            }
            
            # Query registry
            return self.registry.find_signals(criteria)
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'timeframe': timeframe},
                subsystem='signal',
                component='SignalOrchestrator'
            )
            return []
    
    def get_signals_by_symbol(self, symbol: str, context: Dict[str, Any]) -> List[Signal]:
        """
        Get signals for a specific symbol
        
        Args:
            symbol: Trading symbol
            context: Query context
            
        Returns:
            List of signals for the symbol
        """
        try:
            if not self.registry:
                return []
            
            # Build search criteria
            criteria = {
                'symbol': symbol,
                'active_only': context.get('active_only', True),
                **{k: v for k, v in context.items() if k not in ['active_only']}
            }
            
            # Query registry
            return self.registry.find_signals(criteria)
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'symbol': symbol},
                subsystem='signal',
                component='SignalOrchestrator'
            )
            return []
    
    def resolve_conflicting_signals(self, signals: List[Signal], 
                                 context: Dict[str, Any]) -> List[Signal]:
        """
        Resolve conflicts between signals
        
        This method resolves conflicts between signals for the same symbol
        and timeframe based on priority, confidence, and strength.
        
        Args:
            signals: List of potentially conflicting signals
            context: Resolution context
            
        Returns:
            List of resolved signals
        """
        try:
            if not signals:
                return []
            
            # Group signals by symbol and timeframe
            grouped: Dict[Tuple[str, str], List[Signal]] = {}
            for signal in signals:
                key = (signal.symbol, signal.timeframe.value)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(signal)
            
            # Resolve conflicts in each group
            resolved = []
            for (symbol, timeframe), group in grouped.items():
                # If only one signal in the group, no conflict to resolve
                if len(group) == 1:
                    resolved.append(group[0])
                    continue
                
                # Look for bullish and bearish signals
                bullish = [s for s in group if s.direction == SignalDirection.BULLISH]
                bearish = [s for s in group if s.direction == SignalDirection.BEARISH]
                neutral = [s for s in group if s.direction == SignalDirection.NEUTRAL]
                
                # No direction conflict if all signals have the same direction
                if not bullish or not bearish:
                    # Add signals in order of priority, confidence, and strength
                    sorted_signals = sorted(
                        group,
                        key=lambda s: (
                            s.metadata.priority.value,
                            s.metadata.confidence.value,
                            s.strength.value
                        ),
                        reverse=True
                    )
                    resolved.append(sorted_signals[0])
                    continue
                
                # Calculate weighted scores for bullish and bearish signals
                bullish_score = sum(
                    s.metadata.priority.value * s.metadata.confidence.value * s.strength.value
                    for s in bullish
                )
                
                bearish_score = sum(
                    s.metadata.priority.value * s.metadata.confidence.value * s.strength.value
                    for s in bearish
                )
                
                # Determine the winning direction
                if bullish_score > bearish_score:
                    # Bullish wins, add top bullish signal
                    top_bullish = sorted(
                        bullish,
                        key=lambda s: (
                            s.metadata.priority.value,
                            s.metadata.confidence.value,
                            s.strength.value
                        ),
                        reverse=True
                    )[0]
                    resolved.append(top_bullish)
                elif bearish_score > bullish_score:
                    # Bearish wins, add top bearish signal
                    top_bearish = sorted(
                        bearish,
                        key=lambda s: (
                            s.metadata.priority.value,
                            s.metadata.confidence.value,
                            s.strength.value
                        ),
                        reverse=True
                    )[0]
                    resolved.append(top_bearish)
                else:
                    # Scores are equal, add a neutral signal if available, otherwise
                    # add both top bullish and top bearish signals
                    if neutral:
                        top_neutral = sorted(
                            neutral,
                            key=lambda s: (
                                s.metadata.priority.value,
                                s.metadata.confidence.value,
                                s.strength.value
                            ),
                            reverse=True
                        )[0]
                        resolved.append(top_neutral)
                    else:
                        # Add both top signals
                        top_bullish = sorted(
                            bullish,
                            key=lambda s: (
                                s.metadata.priority.value,
                                s.metadata.confidence.value,
                                s.strength.value
                            ),
                            reverse=True
                        )[0]
                        top_bearish = sorted(
                            bearish,
                            key=lambda s: (
                                s.metadata.priority.value,
                                s.metadata.confidence.value,
                                s.strength.value
                            ),
                            reverse=True
                        )[0]
                        resolved.append(top_bullish)
                        resolved.append(top_bearish)
            
            return resolved
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'signal_count': len(signals)},
                subsystem='signal',
                component='SignalOrchestrator'
            )
            return signals  # Return original signals on error
    
    def cascade_timeframe_signals(self, signal: Signal, 
                               context: Dict[str, Any]) -> List[Signal]:
        """
        Cascade a signal to lower timeframes
        
        This method creates derived signals for lower timeframes based
        on a higher timeframe signal, with decreasing strength.
        
        Args:
            signal: Source signal from higher timeframe
            context: Cascading context
            
        Returns:
            List of cascaded signals for lower timeframes
        """
        try:
            # Get timeframe priority
            timeframe_priority = self.parameters.get('timeframe_priority', 
                                                   ['1M', '1w', '1d', '4h', '1h', '30m', '15m', '5m', '1m'])
            
            # Find source timeframe index
            source_tf = signal.timeframe.value
            if source_tf not in timeframe_priority:
                self.logger.warning(f"Timeframe {source_tf} not found in priority list")
                return []
            
            source_index = timeframe_priority.index(source_tf)
            
            # If already at lowest timeframe, nothing to cascade
            if source_index >= len(timeframe_priority) - 1:
                return []
            
            # Create cascaded signals for lower timeframes
            cascaded = []
            strength_factor = context.get('strength_factor', 0.8)  # Strength decreases by 20% per level
            
            for i in range(source_index + 1, len(timeframe_priority)):
                target_tf = timeframe_priority[i]
                
                # Calculate cascaded strength
                level_diff = i - source_index
                strength_value = max(1, int(signal.strength.value * (strength_factor ** level_diff)))
                
                # Find appropriate strength enum value
                cascaded_strength = None
                for strength in SignalStrength:
                    if strength.value == strength_value:
                        cascaded_strength = strength
                        break
                
                if cascaded_strength is None:
                    if strength_value >= SignalStrength.VERY_STRONG.value:
                        cascaded_strength = SignalStrength.VERY_STRONG
                    elif strength_value >= SignalStrength.STRONG.value:
                        cascaded_strength = SignalStrength.STRONG
                    elif strength_value >= SignalStrength.MODERATE.value:
                        cascaded_strength = SignalStrength.MODERATE
                    else:
                        cascaded_strength = SignalStrength.WEAK
                
                # Create cascaded signal metadata
                cascaded_metadata = copy.deepcopy(signal.metadata)
                cascaded_metadata.source = f"Cascade({signal.metadata.source})"
                
                # Add cascaded relationship tag
                if 'cascaded' not in cascaded_metadata.tags:
                    cascaded_metadata.tags.append('cascaded')
                
                # Create cascaded signal
                cascaded_signal = copy.deepcopy(signal)
                cascaded_signal.id = f"{signal.id}_cascade_{target_tf}"
                cascaded_signal.timeframe = next(tf for tf in Timeframe if tf.value == target_tf)
                cascaded_signal.strength = cascaded_strength
                cascaded_signal.metadata = cascaded_metadata
                cascaded_signal.parent_signals = [signal.id]
                cascaded_signal.child_signals = []
                
                cascaded.append(cascaded_signal)
            
            return cascaded
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'signal_id': signal.id},
                subsystem='signal',
                component='SignalOrchestrator'
            )
            return []
    
    def _process_generated_signals(self, signals: List[Signal], 
                                 context: Dict[str, Any]) -> List[Signal]:
        """
        Process signals after generation
        
        This method applies common processing to signals after they are generated,
        such as linking to parents, validating, etc.
        
        Args:
            signals: Newly generated signals
            context: Processing context
            
        Returns:
            Processed signals
        """
        if not signals:
            return []
        
        # Apply filters if specified
        if 'post_generation_filters' in context:
            filter_names = context['post_generation_filters']
            for name in filter_names:
                if name not in self.filters:
                    continue
                
                filter_component = self.filters[name]
                signals = filter_component.filter_signals(signals, context)
        
        # Apply processors if specified
        if 'post_generation_processors' in context:
            processor_names = context['post_generation_processors']
            for name in processor_names:
                if name not in self.processors:
                    continue
                
                processor = self.processors[name]
                signals = processor.process_signals(signals, context)
        
        return signals
    
    def _load_pipeline_config(self, config_path: str) -> None:
        """
        Load pipeline configurations from a file
        
        Args:
            config_path: Path to config file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create pipelines from config
            if 'pipelines' in config and isinstance(config['pipelines'], list):
                for pipeline_config in config['pipelines']:
                    if 'id' in pipeline_config:
                        self.pipelines[pipeline_config['id']] = pipeline_config
                        self.logger.info(f"Loaded pipeline configuration: {pipeline_config['id']}")
            
            self.logger.info(f"Loaded {len(self.pipelines)} pipeline configurations from {config_path}")
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'config_path': config_path},
                subsystem='signal',
                component='SignalOrchestrator'
            )
    
    def _register_event_handlers(self) -> None:
        """Register event handlers for system events"""
        # Register handler for data updates
        event_bus.subscribe('data_pipeline_update', self._handle_data_pipeline_update)
        
        # Register handler for signal expiration
        event_bus.subscribe('signal_expired', self._handle_signal_expired)
    
    def _handle_data_pipeline_update(self, data: Dict[str, Any]) -> None:
        """
        Handle data pipeline update event
        
        Args:
            data: Event data with pipeline results
        """
        try:
            pipeline_data = data.get('data')
            context = data.get('context', {})
            
            if pipeline_data:
                self.handle_data_update(pipeline_data, context)
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'event': 'data_pipeline_update'},
                subsystem='signal',
                component='SignalOrchestrator'
            )
    
    def _handle_signal_expired(self, data: Dict[str, Any]) -> None:
        """
        Handle signal expired event
        
        Args:
            data: Event data with signal ID
        """
        try:
            signal_id = data.get('signal_id')
            
            if signal_id and self.registry:
                self.registry.remove_signal(signal_id)
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'event': 'signal_expired'},
                subsystem='signal',
                component='SignalOrchestrator'
            )