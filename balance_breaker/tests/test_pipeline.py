"""
Test the full data pipeline functionality
"""

import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

from balance_breaker.src.data_pipeline import (
    DataPipelineOrchestrator,
    PriceLoader,
    DataNormalizer,
    TimeAligner,
    TechnicalIndicators
)

def test_pipeline():
    print("Testing full data pipeline...")
    
    # 1. Create the orchestrator
    orchestrator = DataPipelineOrchestrator()
    print("✓ Created orchestrator")
    
    # 2. Register components
    # Loaders
    orchestrator.register_component(PriceLoader({'repository_path': 'balance_breaker/data/price'}))
    
    # Processors
    orchestrator.register_component(DataNormalizer())
    orchestrator.register_component(TimeAligner())
    
    # Indicators
    orchestrator.register_component(TechnicalIndicators({
        'sma_periods': [20, 50],
        'rsi_period': 14
    }))
    
    print("✓ Registered components")
    
    # 3. Define request
    request = {
        'pairs': ['EURUSD'],
        'start_date': '2023-01-01',
        'end_date': '2023-02-01',
        'data_type': 'price',
        'align': True,
        'indicators': ['TechnicalIndicators']
    }
    
    # 4. Create and execute pipeline
    print("Creating pipeline...")
    pipeline = orchestrator.create_pipeline(request)
    
    if not pipeline:
        print("❌ Failed to create pipeline")
        return
    
    print(f"✓ Created pipeline with {len(pipeline)} components:")
    for i, component in enumerate(pipeline):
        print(f"  {i+1}. {component.name} ({component.component_type})")
    
    print("\nExecuting pipeline...")
    try:
        result = orchestrator.execute_pipeline(pipeline, request)
        
        # 5. Check the results
        if result and 'price' in result and 'EURUSD' in result['price']:
            df = result['price']['EURUSD']
            print("\n✓ Pipeline execution successful!")
            print(f"Loaded {len(df)} rows of data for EURUSD")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Columns: {', '.join(df.columns)}")
            
            # Print the first few rows
            print("\nFirst 5 rows:")
            print(df.head(5))
            
            # Print some statistics
            print("\nData statistics:")
            print(df[['close', 'SMA_20', 'RSI']].describe())
            
        else:
            print("❌ Pipeline returned unexpected result structure")
            pprint(result)
    
    except Exception as e:
        print(f"❌ Error executing pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()