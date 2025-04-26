"""
Simple Data Pipeline Test

Tests basic pipeline functionality with price data loading.
"""

import logging
logging.basicConfig(level=logging.INFO)

# Import key components
from balance_breaker.src.data_pipeline import (
    DataPipelineOrchestrator,
    PriceLoader
)

def test_basic_pipeline():
    """Test basic pipeline functionality"""
    print("Creating orchestrator...")
    orchestrator = DataPipelineOrchestrator()
    
    print("Registering PriceLoader...")
    price_loader = PriceLoader()
    orchestrator.register_component(price_loader)
    
    print("Creating pipeline...")
    request = {
        'pairs': ['EURUSD'],
        'start_date': '2023-01-01',
        'end_date': '2023-01-31',
        'data_type': 'price'
    }
    
    pipeline = orchestrator.create_pipeline(request)
    print(f"Pipeline created with {len(pipeline)} components")
    
    print("Pipeline components:")
    for comp in pipeline:
        print(f"- {comp.name} ({comp.component_type})")
    
    print("\nTest successful!")

if __name__ == "__main__":
    test_basic_pipeline()