"""
Data Pipeline Orchestrator

Central component that coordinates the data pipeline system, managing the flow
of data through various processing stages and components.
"""

import logging
import threading
import json
import os
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from datetime import datetime

from balance_breaker.src.core.interface_registry import implements, registry
from balance_breaker.src.core.error_handling import ErrorHandler, BalanceBreakerError, ErrorSeverity, ErrorCategory
from balance_breaker.src.core.parameter_manager import ParameterizedComponent
from balance_breaker.src.core.integration_tools import event_bus, integrates_with, IntegrationType
from balance_breaker.src.data_pipeline.data_pipeline_interfaces import (
    PipelineComponent, DataLoader, DataProcessor, DataValidator,
    DataAligner, IndicatorCalculator, DataSerializer, DataRegistry
)


class DataPipelineError(BalanceBreakerError):
    """Error in data pipeline operations"""
    
    def __init__(self, 
                message: str, 
                component: str = "",
                severity: ErrorSeverity = ErrorSeverity.ERROR,
                context: Optional[Dict[str, Any]] = None,
                original_exception: Optional[Exception] = None):
        super().__init__(
            message=message, 
            subsystem="data_pipeline",
            component=component,
            severity=severity,
            category=ErrorCategory.EXECUTION,
            context=context,
            original_exception=original_exception
        )


class DataPipelineOrchestrator(ParameterizedComponent):
    """
    Central orchestrator for data pipeline operations
    
    This orchestrator manages the flow of data through loaders, validators,
    processors, aligners, indicators, and serializers. It supports both
    request-based and configuration-driven pipeline creation.
    
    Parameters:
    -----------
    auto_register_data : bool
        Whether to automatically register processed data (default: True)
    enable_event_processing : bool
        Whether to enable event-based data processing (default: True)
    registry_id : str
        ID of the data registry to use (default: 'default')
    pipeline_config_path : str
        Path to pipeline configuration file (default: None)
    """
    
    def __init__(self, parameters=None):
        """Initialize data pipeline orchestrator with optional parameters"""
        # Define default parameters
        default_params = {
            'auto_register_data': True,
            'enable_event_processing': True,
            'registry_id': 'default',
            'pipeline_config_path': None
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
        
        # Set up logging and error handling
        self.logger = logging.getLogger(__name__)
        self.error_handler = ErrorHandler(self.logger)
        
        # Component registries by type
        self.loaders: Dict[str, DataLoader] = {}
        self.validators: Dict[str, DataValidator] = {}
        self.processors: Dict[str, DataProcessor] = {}
        self.aligners: Dict[str, DataAligner] = {}
        self.indicators: Dict[str, IndicatorCalculator] = {}
        self.serializers: Dict[str, DataSerializer] = {}
        
        # Data registry reference
        self.registry_id = self.parameters.get('registry_id', 'default')
        self.registry: Optional[DataRegistry] = None
        
        # Pipeline configurations
        self.pipelines: Dict[str, Dict[str, Any]] = {}
        
        # Execution metrics
        self.metrics: Dict[str, Dict[str, float]] = {}
        
        # Thread lock
        self._lock = threading.RLock()
        
        # Load pipeline configurations if provided
        pipeline_config_path = self.parameters.get('pipeline_config_path')
        if pipeline_config_path and os.path.exists(pipeline_config_path):
            self._load_pipeline_config(pipeline_config_path)
        
        # Register event handlers if enabled
        if self.parameters.get('enable_event_processing', True):
            self._register_event_handlers()
    
    def set_registry(self, registry: DataRegistry) -> None:
        """
        Set the data registry
        
        Args:
            registry: Data registry to use
        """
        self.registry = registry
    
    def register_loader(self, loader: DataLoader) -> bool:
        """
        Register a data loader
        
        Args:
            loader: Data loader to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            # Validate loader implements DataLoader interface
            validation = registry.validate_implementation(loader, "DataLoader")
            if not validation.get('valid', False):
                self.logger.error(f"Invalid data loader: {validation}")
                return False
            
            with self._lock:
                name = loader.name
                self.loaders[name] = loader
                self.logger.info(f"Registered data loader: {name}")
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'loader_name': getattr(loader, 'name', None)},
                subsystem='data_pipeline',
                component='DataPipelineOrchestrator'
            )
            return False
    
    def register_validator(self, validator: DataValidator) -> bool:
        """
        Register a data validator
        
        Args:
            validator: Data validator to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            # Validate validator implements DataValidator interface
            validation = registry.validate_implementation(validator, "DataValidator")
            if not validation.get('valid', False):
                self.logger.error(f"Invalid data validator: {validation}")
                return False
            
            with self._lock:
                name = validator.name
                self.validators[name] = validator
                self.logger.info(f"Registered data validator: {name}")
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'validator_name': getattr(validator, 'name', None)},
                subsystem='data_pipeline',
                component='DataPipelineOrchestrator'
            )
            return False
    
    def register_processor(self, processor: DataProcessor) -> bool:
        """
        Register a data processor
        
        Args:
            processor: Data processor to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            # Validate processor implements DataProcessor interface
            validation = registry.validate_implementation(processor, "DataProcessor")
            if not validation.get('valid', False):
                self.logger.error(f"Invalid data processor: {validation}")
                return False
            
            with self._lock:
                name = processor.name
                self.processors[name] = processor
                self.logger.info(f"Registered data processor: {name}")
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'processor_name': getattr(processor, 'name', None)},
                subsystem='data_pipeline',
                component='DataPipelineOrchestrator'
            )
            return False
    
    def register_aligner(self, aligner: DataAligner) -> bool:
        """
        Register a data aligner
        
        Args:
            aligner: Data aligner to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            # Validate aligner implements DataAligner interface
            validation = registry.validate_implementation(aligner, "DataAligner")
            if not validation.get('valid', False):
                self.logger.error(f"Invalid data aligner: {validation}")
                return False
            
            with self._lock:
                name = aligner.name
                self.aligners[name] = aligner
                self.logger.info(f"Registered data aligner: {name}")
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'aligner_name': getattr(aligner, 'name', None)},
                subsystem='data_pipeline',
                component='DataPipelineOrchestrator'
            )
            return False
    
    def register_indicator(self, indicator: IndicatorCalculator) -> bool:
        """
        Register an indicator calculator
        
        Args:
            indicator: Indicator calculator to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            # Validate indicator implements IndicatorCalculator interface
            validation = registry.validate_implementation(indicator, "IndicatorCalculator")
            if not validation.get('valid', False):
                self.logger.error(f"Invalid indicator calculator: {validation}")
                return False
            
            with self._lock:
                name = indicator.name
                self.indicators[name] = indicator
                self.logger.info(f"Registered indicator calculator: {name}")
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'indicator_name': getattr(indicator, 'name', None)},
                subsystem='data_pipeline',
                component='DataPipelineOrchestrator'
            )
            return False
    
    def register_serializer(self, serializer: DataSerializer) -> bool:
        """
        Register a data serializer
        
        Args:
            serializer: Data serializer to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            # Validate serializer implements DataSerializer interface
            validation = registry.validate_implementation(serializer, "DataSerializer")
            if not validation.get('valid', False):
                self.logger.error(f"Invalid data serializer: {validation}")
                return False
            
            with self._lock:
                name = serializer.name
                self.serializers[name] = serializer
                self.logger.info(f"Registered data serializer: {name}")
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'serializer_name': getattr(serializer, 'name', None)},
                subsystem='data_pipeline',
                component='DataPipelineOrchestrator'
            )
            return False
    
    def register_component(self, component: PipelineComponent) -> bool:
        """
        Register a pipeline component based on its type
        
        Args:
            component: Component to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        component_type = component.component_type
        
        if component_type == 'loader':
            return self.register_loader(component)
        elif component_type == 'validator':
            return self.register_validator(component)
        elif component_type == 'processor':
            return self.register_processor(component)
        elif component_type == 'aligner':
            return self.register_aligner(component)
        elif component_type == 'indicator':
            return self.register_indicator(component)
        elif component_type == 'serializer':
            return self.register_serializer(component)
        else:
            self.logger.error(f"Unknown component type: {component_type}")
            return False
    
    def register_modular_indicators(self) -> int:
        """
        Register all available modular indicators
        
        Returns:
            Number of indicators registered
        """
        try:
            from balance_breaker.src.data_pipeline.indicators.modular_base import indicator_registry
            
            # Get all available indicators from registry
            indicator_names = indicator_registry.list_all_indicators()
            
            registered_count = 0
            for name in indicator_names:
                indicator_class = indicator_registry.get_indicator_by_name(name)
                if indicator_class:
                    # Create indicator instance
                    indicator = indicator_class()
                    # Register with orchestrator
                    if self.register_indicator(indicator):
                        registered_count += 1
            
            self.logger.info(f"Registered {registered_count} modular indicators")
            return registered_count
            
        except ImportError:
            self.logger.warning("Modular indicator system not available")
            return 0
    
    def create_pipeline(self, config: Dict[str, Any]) -> str:
        """
        Create a data processing pipeline
        
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
                
                self.logger.info(f"Created data pipeline: {pipeline_id}")
                return pipeline_id
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'config': config},
                subsystem='data_pipeline',
                component='DataPipelineOrchestrator'
            )
            raise DataPipelineError(
                message=f"Failed to create pipeline: {str(e)}",
                component='DataPipelineOrchestrator',
                context={'config': config},
                original_exception=e
            )
    
    def get_pipeline_components(self, pipeline_id: str) -> List[PipelineComponent]:
        """
        Get the components for a pipeline
        
        Args:
            pipeline_id: ID of the pipeline
            
        Returns:
            List of pipeline components
        """
        if pipeline_id not in self.pipelines:
            raise DataPipelineError(
                message=f"Pipeline not found: {pipeline_id}",
                component='DataPipelineOrchestrator'
            )
        
        pipeline_config = self.pipelines[pipeline_id]
        components = []
        
        # Add loaders
        loader_names = pipeline_config.get('loaders', [])
        for name in loader_names:
            if name in self.loaders:
                components.append(self.loaders[name])
        
        # Add validators
        validator_names = pipeline_config.get('validators', [])
        for name in validator_names:
            if name in self.validators:
                components.append(self.validators[name])
        
        # Add processors
        processor_names = pipeline_config.get('processors', [])
        for name in processor_names:
            if name in self.processors:
                components.append(self.processors[name])
        
        # Add aligners
        aligner_names = pipeline_config.get('aligners', [])
        for name in aligner_names:
            if name in self.aligners:
                components.append(self.aligners[name])
        
        # Add indicators
        indicator_names = pipeline_config.get('indicators', [])
        for name in indicator_names:
            if name in self.indicators:
                components.append(self.indicators[name])
        
        # Add serializers
        serializer_names = pipeline_config.get('serializers', [])
        for name in serializer_names:
            if name in self.serializers:
                components.append(self.serializers[name])
        
        return components
    
    def legacy_create_pipeline(self, request: Dict[str, Any]) -> List[PipelineComponent]:
        """
        Create a pipeline based on request parameters (legacy mode)
        
        This method maintains backward compatibility with the original
        pipeline creation approach.
        
        Args:
            request: Request parameters
            
        Returns:
            List of pipeline components to execute
        """
        pipeline = []
        
        # Determine which components to use based on request
        data_type = request.get('data_type', 'price')
        
        # 1. Add appropriate loader
        if data_type == 'price' and 'PriceLoader' in self.loaders:
            pipeline.append(self.loaders['PriceLoader'])
        elif data_type == 'macro' and 'MacroLoader' in self.loaders:
            pipeline.append(self.loaders['MacroLoader'])
        elif 'DataLoader' in self.loaders:
            pipeline.append(self.loaders['DataLoader'])
        
        # 2. Add validator if available
        if 'DataValidator' in self.validators:
            pipeline.append(self.validators['DataValidator'])
        
        # 3. Add data processor for normalization
        if 'DataNormalizer' in self.processors:
            pipeline.append(self.processors['DataNormalizer'])
        
        # 4. Add gap detector if requested
        if request.get('detect_gaps', False) and 'GapDetector' in self.validators:
            pipeline.append(self.validators['GapDetector'])
        
        # 5. Add data aligner
        if request.get('align', True) and 'TimeAligner' in self.aligners:
            pipeline.append(self.aligners['TimeAligner'])
        
        # 6. Add indicators if requested
        indicators = request.get('indicators', [])
        if indicators:
            for indicator_name in indicators:
                if indicator_name in self.indicators:
                    pipeline.append(self.indicators[indicator_name])
        
        # 7. Add serializer if needed
        if request.get('export', False) and 'DataExporter' in self.serializers:
            pipeline.append(self.serializers['DataExporter'])
        
        # Convert request to pipeline configuration for future use
        pipeline_config = {
            'id': f"legacy_{time.time()}",
            'loaders': [c.name for c in pipeline if c.component_type == 'loader'],
            'validators': [c.name for c in pipeline if c.component_type == 'validator'],
            'processors': [c.name for c in pipeline if c.component_type == 'processor'],
            'aligners': [c.name for c in pipeline if c.component_type == 'aligner'],
            'indicators': [c.name for c in pipeline if c.component_type == 'indicator'],
            'serializers': [c.name for c in pipeline if c.component_type == 'serializer'],
            'context': {**request}
        }
        
        # Store pipeline configuration
        self.pipelines[pipeline_config['id']] = pipeline_config
        
        return pipeline
    
    def execute_pipeline(self, pipeline_id_or_components: Union[str, List[PipelineComponent]], 
                        context: Dict[str, Any]) -> Any:
        """
        Execute a pipeline and return the results
        
        Args:
            pipeline_id_or_components: Pipeline ID or list of components
            context: Execution context
            
        Returns:
            Processed data
        """
        try:
            # Determine pipeline components
            if isinstance(pipeline_id_or_components, str):
                # Get components for pipeline ID
                pipeline_id = pipeline_id_or_components
                components = self.get_pipeline_components(pipeline_id)
                pipeline_config = self.pipelines[pipeline_id]
            else:
                # Use provided components
                components = pipeline_id_or_components
                pipeline_id = f"dynamic_{time.time()}"
                pipeline_config = None
            
            # Check for cached result if registry is available
            cache_key = None
            if self.registry and context.get('use_cache', True):
                # Generate cache key from context
                cache_key = self._generate_cache_key(context)
                # Look for cached data
                cached_results = self.registry.find_data({
                    'tags': ['cache'],
                    'cache_key': cache_key,
                    'active_only': True
                })
                if cached_results:
                    self.logger.info(f"Cache hit for key: {cache_key}")
                    return cached_results[0]['data']
            
            # Track execution metrics
            start_time = datetime.now()
            pipeline_metrics = {}
            
            # Execute pipeline components in sequence
            data = None
            
            for component in components:
                component_start = time.time()
                self.logger.debug(f"Executing component: {component.name}")
                
                try:
                    # Process data through this component
                    data = component.process(data, context)
                    
                    # Record metrics
                    component_end = time.time()
                    execution_time = component_end - component_start
                    pipeline_metrics[component.name] = execution_time
                    
                    self.logger.debug(f"Completed {component.name} in {execution_time:.4f} seconds")
                    
                except Exception as e:
                    # Handle component error
                    self.logger.error(f"Error in component {component.name}: {str(e)}")
                    
                    # Create detailed error
                    error = DataPipelineError(
                        message=f"Pipeline error in {component.name}: {str(e)}",
                        component=component.name,
                        context=context,
                        original_exception=e
                    )
                    
                    # Add error handling logic here (retry, fallback, etc.)
                    self._handle_error(error, component, context)
                    
                    # Re-raise the error
                    raise error
            
            # Calculate overall metrics
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            # Store pipeline metrics
            with self._lock:
                self.metrics[pipeline_id] = {
                    'components': pipeline_metrics,
                    'total_duration': total_duration,
                    'timestamp': datetime.now()
                }
            
            # Cache result if enabled and registry is available
            if self.registry and context.get('use_cache', True) and data is not None:
                # Only cache if result is not None
                try:
                    # Prepare metadata
                    cache_metadata = {
                        'data_type': 'cache',
                        'source': 'pipeline',
                        'pipeline_id': pipeline_id,
                        'tags': ['cache'],
                        'cache_key': cache_key,
                        # Set expiration based on TTL
                        'expiration_time': datetime.now() + \
                            timedelta(seconds=context.get('cache_ttl', 3600))
                    }
                    
                    # Register data in registry
                    self.registry.register_data(data, cache_metadata)
                    
                except Exception as e:
                    # Log but don't fail on cache error
                    self.logger.warning(f"Failed to cache pipeline result: {str(e)}")
            
            # Register results if auto-register is enabled
            auto_register = (
                (pipeline_config and pipeline_config.get('auto_register_data', False)) or
                self.parameters.get('auto_register_data', True)
            )
            
            if auto_register and self.registry and data is not None:
                # Prepare metadata
                data_metadata = {
                    'data_type': context.get('data_type', 'unknown'),
                    'source': 'pipeline',
                    'pipeline_id': pipeline_id,
                    'pairs': context.get('pairs', []),
                    'timeframe': context.get('timeframe'),
                    'start_date': context.get('start_date'),
                    'end_date': context.get('end_date'),
                    'tags': context.get('tags', []) + ['pipeline_result']
                }
                
                # Register data in registry
                data_id = self.registry.register_data(data, data_metadata)
                
                # Update context with data ID
                context['data_id'] = data_id
            
            # Publish pipeline execution event
            event_bus.publish('pipeline_executed', {
                'pipeline_id': pipeline_id,
                'duration': total_duration,
                'component_count': len(components)
            })
            
            return data
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'pipeline_id': pipeline_id_or_components 
                        if isinstance(pipeline_id_or_components, str) else 'dynamic'},
                subsystem='data_pipeline',
                component='DataPipelineOrchestrator'
            )
            raise DataPipelineError(
                message=f"Failed to execute pipeline: {str(e)}",
                component='DataPipelineOrchestrator',
                context={'pipeline_id': pipeline_id_or_components 
                       if isinstance(pipeline_id_or_components, str) else 'dynamic'},
                original_exception=e
            )
    
    @integrates_with(
        target_subsystem='signal',
        integration_type=IntegrationType.DATA_FLOW,
        description='Provides data for signal generation'
    )
    def get_data_for_signals(self, request: Dict[str, Any]) -> Any:
        """
        Get data specifically formatted for signal generation
        
        Args:
            request: Data request parameters
            
        Returns:
            Processed data ready for signal generation
        """
        # Include signal formatting flag in context
        context = {**request, 'format_for_signals': True}
        
        # Create a pipeline config optimized for signal generation
        config = {
            'id': f"signal_pipeline_{time.time()}",
            'loaders': ['PriceLoader'] if 'price' in request.get('data_type', 'price') else ['MacroLoader'],
            'validators': ['DataValidator'],
            'processors': ['DataNormalizer'],
            'aligners': ['TimeAligner'] if request.get('align', True) else [],
            'indicators': request.get('indicators', []),
            'context': context
        }
        
        # Create and execute pipeline
        pipeline_id = self.create_pipeline(config)
        return self.execute_pipeline(pipeline_id, context)
    
    def get_data(self, pairs: List[str], start_date: str, end_date: str, 
                data_type: str = 'price', indicators: List[str] = None, 
                options: Dict[str, Any] = None) -> Any:
        """
        Get processed data for the specified parameters (legacy API)
        
        This method maintains backward compatibility with the original API.
        
        Args:
            pairs: List of currency pairs
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data_type: Type of data ('price' or 'macro')
            indicators: List of indicators to calculate
            options: Additional options
            
        Returns:
            Processed data
        """
        # Create request context
        context = {
            'pairs': pairs,
            'start_date': start_date,
            'end_date': end_date,
            'data_type': data_type,
            'indicators': indicators or [],
            **(options or {})
        }
        
        # Check if we have a cached pipeline for this request pattern
        pipeline_key = f"{data_type}_{','.join(sorted(indicators or []))}"
        
        pipeline_id = None
        for pid, config in self.pipelines.items():
            if config.get('cache_key') == pipeline_key:
                pipeline_id = pid
                break
        
        if not pipeline_id:
            # Create a new pipeline
            components = self.legacy_create_pipeline(context)
            pipeline_id = f"legacy_{time.time()}"
            self.pipelines[pipeline_id] = {
                'id': pipeline_id,
                'cache_key': pipeline_key,
                'components': [c.name for c in components],
                'context': {}
            }
        
        # Execute the pipeline
        return self.execute_pipeline(pipeline_id, context)
    
    def handle_data_update(self, data_type: str, update_context: Dict[str, Any]) -> Any:
        """
        Handle data update event
        
        This method is called when new data is available. It finds and executes
        pipelines that match the data type.
        
        Args:
            data_type: Type of updated data
            update_context: Update context
            
        Returns:
            Processing results
        """
        try:
            results = {}
            
            # Find pipelines that match the data type
            matching_pipelines = []
            for pipeline_id, config in self.pipelines.items():
                if config.get('data_types', []):
                    if data_type in config['data_types']:
                        matching_pipelines.append(pipeline_id)
                elif config.get('auto_execute', False):
                    matching_pipelines.append(pipeline_id)
            
            # Execute matching pipelines
            for pipeline_id in matching_pipelines:
                try:
                    # Create merged context
                    context = {
                        'data_type': data_type,
                        'update_time': datetime.now(),
                        **update_context
                    }
                    
                    # Execute pipeline
                    result = self.execute_pipeline(pipeline_id, context)
                    results[pipeline_id] = result
                    
                except Exception as e:
                    self.error_handler.handle_error(
                        e,
                        context={'pipeline_id': pipeline_id, 'data_type': data_type},
                        subsystem='data_pipeline',
                        component='DataPipelineOrchestrator'
                    )
            
            # Publish data processing event
            event_bus.publish('data_processed', {
                'data_type': data_type,
                'pipelines': list(results.keys())
            })
            
            return results
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'data_type': data_type},
                subsystem='data_pipeline',
                component='DataPipelineOrchestrator'
            )
            return {}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get pipeline performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        with self._lock:
            return {
                'pipelines': self.metrics,
                'components': self._aggregate_component_metrics()
            }
    
    def clear_metrics(self) -> None:
        """Clear performance metrics"""
        with self._lock:
            self.metrics = {}
    
    def _aggregate_component_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics by component
        
        Returns:
            Dictionary of component metrics
        """
        component_metrics = {}
        
        for pipeline_metrics in self.metrics.values():
            component_times = pipeline_metrics.get('components', {})
            for component_name, execution_time in component_times.items():
                if component_name not in component_metrics:
                    component_metrics[component_name] = {
                        'total_time': 0.0,
                        'call_count': 0,
                        'avg_time': 0.0
                    }
                
                component_metrics[component_name]['total_time'] += execution_time
                component_metrics[component_name]['call_count'] += 1
        
        # Calculate averages
        for metrics in component_metrics.values():
            if metrics['call_count'] > 0:
                metrics['avg_time'] = metrics['total_time'] / metrics['call_count']
        
        return component_metrics
    
    def _generate_cache_key(self, context: Dict[str, Any]) -> str:
        """
        Generate a cache key from context
        
        Args:
            context: Pipeline context
            
        Returns:
            Cache key string
        """
        # Create a serializable subset of the context
        cache_dict = {
            'pairs': sorted(context.get('pairs', [])),
            'start_date': context.get('start_date', ''),
            'end_date': context.get('end_date', ''),
            'data_type': context.get('data_type', ''),
            'indicators': sorted(context.get('indicators', [])),
            'timeframe': context.get('timeframe', '')
        }
        
        # Add any other cacheable keys
        for key, value in context.items():
            if key not in cache_dict and isinstance(value, (str, bool, int, float)):
                cache_dict[key] = value
        
        # Convert to JSON and hash
        import hashlib
        json_str = json.dumps(cache_dict, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
    
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
                subsystem='data_pipeline',
                component='DataPipelineOrchestrator'
            )
    
    def _register_event_handlers(self) -> None:
        """Register event handlers for system events"""
        # Register handler for data updates
        event_bus.subscribe('new_data_available', self._handle_new_data_event)
    
    def _handle_new_data_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle new data available event
        
        Args:
            event_data: Event data
        """
        try:
            data_type = event_data.get('data_type', 'unknown')
            self.handle_data_update(data_type, event_data)
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'event': 'new_data_available'},
                subsystem='data_pipeline',
                component='DataPipelineOrchestrator'
            )
    
    def _handle_error(self, error: DataPipelineError, component: PipelineComponent, context: Dict[str, Any]) -> None:
        """
        Handle pipeline errors
        
        Args:
            error: Pipeline error
            component: Failed component
            context: Pipeline context
        """
        # Log detailed error information
        self.logger.error(f"Pipeline error in component {component.name}: {error.message}")
        
        # Publish error event
        event_bus.publish('pipeline_error', {
            'component': component.name,
            'message': error.message,
            'component_type': component.component_type
        })
        
        # Additional error handling logic can be added here