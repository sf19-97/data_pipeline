"""
Data Pipeline Orchestrator

Central component that coordinates the data pipeline system, managing the flow
of data through various processing stages.
"""

import logging
import hashlib
import json
from typing import Dict, List, Any, Callable, Optional, Type, Union
from abc import ABC, abstractmethod
from datetime import datetime
import time

# Define component interfaces

class PipelineComponent(ABC):
    """Base interface for all pipeline components"""
    
    @abstractmethod
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process data according to component logic
        
        Args:
            data: Input data (can be None for loaders)
            context: Pipeline context information
            
        Returns:
            Processed data
        """
        pass
    
    @property
    @abstractmethod
    def component_type(self) -> str:
        """Return component type identifier"""
        pass
    
    @property
    def name(self) -> str:
        """Return component name"""
        return self.__class__.__name__

class PipelineError(Exception):
    """Base exception for pipeline errors"""
    def __init__(self, message: str, component: Optional[PipelineComponent] = None, stage: str = "", context: Dict[str, Any] = None):
        self.message = message
        self.component = component
        self.stage = stage
        self.context = context or {}
        super().__init__(self.message)

class DataPipelineOrchestrator:
    """Central coordinator for data pipeline operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the orchestrator
        
        Args:
            config: Configuration parameters
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = config or {}
        
        # Component registries by type
        self.components: Dict[str, Dict[str, PipelineComponent]] = {
            'loader': {},
            'validator': {},
            'processor': {},
            'aligner': {},
            'indicator': {},
            'serializer': {}
        }
        
        # Cache for pipeline results
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # Default: 1 hour
        self.cache_enabled = self.config.get('cache_enabled', True)
        
        # Pipeline execution metrics
        self.metrics: Dict[str, Dict[str, float]] = {}
        
        self.logger.info("Data Pipeline Orchestrator initialized")
    
    def register_component(self, component: PipelineComponent) -> None:
        """Register a component with the orchestrator
        
        Args:
            component: Component instance to register
        """
        component_type = component.component_type
        if component_type not in self.components:
            raise ValueError(f"Unknown component type: {component_type}")
        
        self.components[component_type][component.name] = component
        self.logger.debug(f"Registered {component_type} component: {component.name}")
    
    def get_component(self, component_type: str, component_name: str) -> PipelineComponent:
        """Get a registered component
        
        Args:
            component_type: Type of component
            component_name: Name of component
            
        Returns:
            Component instance
        """
        if component_type not in self.components:
            raise ValueError(f"Unknown component type: {component_type}")
        
        if component_name not in self.components[component_type]:
            raise ValueError(f"Component not found: {component_name} of type {component_type}")
        
        return self.components[component_type][component_name]
    
    def create_pipeline(self, request: Dict[str, Any]) -> List[PipelineComponent]:
        """Create a pipeline based on request parameters
        
        Args:
            request: Request parameters
            
        Returns:
            List of pipeline components to execute
        """
        pipeline = []
        
        # Determine which components to use based on request
        data_type = request.get('data_type', 'price')
        
        # 1. Add appropriate loader
        if data_type == 'price':
            if 'PriceLoader' in self.components['loader']:
                pipeline.append(self.components['loader']['PriceLoader'])
        elif data_type == 'macro':
            if 'MacroLoader' in self.components['loader']:
                pipeline.append(self.components['loader']['MacroLoader'])
        else:
            # Default to generic loader
            if 'DataLoader' in self.components['loader']:
                pipeline.append(self.components['loader']['DataLoader'])
        
        # 2. Add validator if available
        if 'DataValidator' in self.components['validator']:
            pipeline.append(self.components['validator']['DataValidator'])
        
        # 3. Add data processor for normalization
        if 'DataNormalizer' in self.components['processor']:
            pipeline.append(self.components['processor']['DataNormalizer'])
        
        # 4. Add gap detector if requested
        if request.get('detect_gaps', False) and 'GapDetector' in self.components['validator']:
            pipeline.append(self.components['validator']['GapDetector'])
        
        # 5. Add data aligner
        if request.get('align', True) and 'TimeAligner' in self.components['aligner']:
            pipeline.append(self.components['aligner']['TimeAligner'])
        
        # 6. Add indicators if requested
        indicators = request.get('indicators', [])
        if indicators:
            # Create a set to track added indicators to avoid duplicates
            added_indicators = set()
            
            for indicator_name in indicators:
                # First, check if the indicator exists directly in our components
                if indicator_name in self.components['indicator']:
                    pipeline.append(self.components['indicator'][indicator_name])
                    added_indicators.add(indicator_name)
                else:
                    # Try to get from indicator registry
                    # This only works if the indicator_registry is imported
                    try:
                        from balance_breaker.src.data_pipeline.indicators.modular_base import indicator_registry
                        indicator_class = indicator_registry.get_indicator_by_name(indicator_name)
                        if indicator_class and indicator_name not in added_indicators:
                            # Create an instance of the indicator
                            indicator = indicator_class()
                            # Register it with orchestrator for future use
                            self.register_component(indicator)
                            # Add to pipeline
                            pipeline.append(indicator)
                            added_indicators.add(indicator_name)
                    except ImportError:
                        # If modular indicators not available, just log a warning
                        self.logger.debug(f"Modular indicator system not available, skipping '{indicator_name}'")
            
            # Handle missing indicators
            missing = [ind for ind in indicators if ind not in added_indicators]
            if missing:
                self.logger.warning(f"Requested indicators not found: {', '.join(missing)}")
        
        # 7. Add serializer if needed
        if request.get('export', False) and 'DataExporter' in self.components['serializer']:
            pipeline.append(self.components['serializer']['DataExporter'])
        
        return pipeline
    
    def register_modular_indicators(self):
        """
        Register all available modular indicators with the orchestrator
        
        Returns:
            Number of indicators registered
        """
        try:
            from balance_breaker.src.data_pipeline.indicators.modular_base import indicator_registry
            
            # Get all available indicators from registry
            indicator_names = indicator_registry.list_all_indicators()
            
            for name in indicator_names:
                indicator_class = indicator_registry.get_indicator_by_name(name)
                if indicator_class:
                    # Create indicator instance
                    indicator = indicator_class()
                    # Register with orchestrator
                    self.register_component(indicator)
                    self.logger.debug(f"Registered indicator: {name}")
            
            return len(indicator_names)
        except ImportError:
            self.logger.warning("Modular indicator system not available")
            return 0
    
    def execute_pipeline(self, pipeline: List[PipelineComponent], request: Dict[str, Any]) -> Any:
        """Execute a pipeline and return the results
        
        Args:
            pipeline: List of pipeline components
            request: Request parameters
            
        Returns:
            Processed data
        """
        # Check cache if enabled
        if self.cache_enabled:
            cache_key = self._generate_cache_key(request)
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                self.logger.info(f"Cache hit for key: {cache_key}")
                return cached_data
        
        # Execute pipeline components in sequence
        data = None
        pipeline_metrics = {}
        
        for component in pipeline:
            start_time = time.time()
            self.logger.debug(f"Executing component: {component.name}")
            
            try:
                # Process data through this component
                data = component.process(data, request)
                
                # Record metrics
                end_time = time.time()
                execution_time = end_time - start_time
                pipeline_metrics[component.name] = execution_time
                
                self.logger.debug(f"Completed {component.name} in {execution_time:.4f} seconds")
                
            except Exception as e:
                # Handle component error
                self.logger.error(f"Error in component {component.name}: {str(e)}")
                
                # Create detailed error
                error = PipelineError(
                    message=f"Pipeline error in {component.name}: {str(e)}",
                    component=component,
                    stage=component.component_type,
                    context=request
                )
                
                # Add error handling logic here (retry, fallback, etc.)
                self._handle_error(error, component, request)
                
                # Re-raise the error
                raise error
        
        # Store pipeline metrics
        pipeline_id = self._generate_pipeline_id(pipeline)
        self.metrics[pipeline_id] = pipeline_metrics
        
        # Cache result if enabled
        if self.cache_enabled:
            self._store_in_cache(cache_key, data)
        
        return data
    
    def get_data(self, pairs: List[str], start_date: str, end_date: str, 
                data_type: str = 'price', indicators: List[str] = None, 
                options: Dict[str, Any] = None) -> Any:
        """Main API: Get processed data for the specified parameters
        
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
        # Create request
        request = {
            'pairs': pairs,
            'start_date': start_date,
            'end_date': end_date,
            'data_type': data_type,
            'indicators': indicators or [],
            'options': options or {}
        }
        
        # Create and execute pipeline
        pipeline = self.create_pipeline(request)
        return self.execute_pipeline(pipeline, request)
    
    def clear_cache(self) -> None:
        """Clear the cache"""
        self.cache = {}
        self.logger.info("Cache cleared")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        return {
            'pipelines': self.metrics,
            'components': self._aggregate_component_metrics()
        }
    
    def _aggregate_component_metrics(self) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics by component
        
        Returns:
            Dictionary of component metrics
        """
        component_metrics = {}
        
        for pipeline_metrics in self.metrics.values():
            for component_name, execution_time in pipeline_metrics.items():
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
    
    def _generate_cache_key(self, request: Dict[str, Any]) -> str:
        """Generate a cache key from request
        
        Args:
            request: Request parameters
            
        Returns:
            Cache key string
        """
        # Create a serializable copy of the request
        cache_dict = {
            'pairs': sorted(request.get('pairs', [])),
            'start_date': request.get('start_date', ''),
            'end_date': request.get('end_date', ''),
            'data_type': request.get('data_type', ''),
            'indicators': sorted(request.get('indicators', [])),
            'options': {k: v for k, v in request.get('options', {}).items()
                      if isinstance(v, (str, bool, int, float))}
        }
        
        # Convert to JSON and hash
        json_str = json.dumps(cache_dict, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def _generate_pipeline_id(self, pipeline: List[PipelineComponent]) -> str:
        """Generate a unique ID for a pipeline
        
        Args:
            pipeline: List of components
            
        Returns:
            Pipeline ID string
        """
        component_names = [component.name for component in pipeline]
        return f"{'-'.join(component_names)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None
        """
        if key not in self.cache:
            return None
        
        cache_entry = self.cache[key]
        timestamp = cache_entry.get('timestamp', 0)
        
        # Check if entry has expired
        if time.time() - timestamp > self.cache_ttl:
            del self.cache[key]
            return None
        
        return cache_entry.get('data')
    
    def _store_in_cache(self, key: str, data: Any) -> None:
        """Store data in cache
        
        Args:
            key: Cache key
            data: Data to cache
        """
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
        self.logger.debug(f"Cached data with key: {key}")
    
    def _handle_error(self, error: PipelineError, component: PipelineComponent, request: Dict[str, Any]) -> None:
        """Handle pipeline errors
        
        Args:
            error: Pipeline error
            component: Failed component
            request: Request parameters
        """
        # Log detailed error information
        self.logger.error(f"Pipeline error in component {component.name}: {error.message}")
        
        # Additional error handling logic can be added here
        # For example:
        # - Retry logic
        # - Fallback mechanisms
        # - Error reporting
        # - Notifications
        
        # For now, we just log the error