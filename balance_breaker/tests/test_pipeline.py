"""
Test Pipeline Functionality

Tests the core data pipeline with actual data processing.
"""

import os
import sys
import pandas as pd
from balance_breaker.src.data_pipeline import (
    DataPipelineOrchestrator,
    PriceLoader, 
    DataNormalizer,
    DataValidator,
    TimeAligner,
    GapDetector
)

def test_full_pipeline():
    print("Testing Balance Breaker Data Pipeline...\n")
    
    # 1. Create orchestrator
    print("1. Creating orchestrator...")
    orchestrator = DataPipelineOrchestrator()
    print("   ✓ Orchestrator created\n")
    
    # 2. Register components
    print("2. Registering components...")
    # Adjust path if needed based on your project structure
    repo_path = os.path.join('balance_breaker', 'data', 'price')
    if not os.path.exists(repo_path):
        print(f"   ! Warning: Repository path {repo_path} does not exist")
        # Try alternative paths
        if os.path.exists('data/price'):
            repo_path = 'data/price'
            print(f"   ✓ Using alternative path: {repo_path}")
    
    price_loader = PriceLoader({'repository_path': repo_path})
    orchestrator.register_component(price_loader)
    print(f"   ✓ Registered PriceLoader with repo path: {repo_path}")
    
    data_validator = DataValidator()
    orchestrator.register_component(data_validator)
    print("   ✓ Registered DataValidator")
    
    gap_detector = GapDetector()
    orchestrator.register_component(gap_detector)
    print("   ✓ Registered GapDetector")
    
    normalizer = DataNormalizer()
    orchestrator.register_component(normalizer)
    print("   ✓ Registered DataNormalizer")
    
    aligner = TimeAligner()
    orchestrator.register_component(aligner)
    print("   ✓ Registered TimeAligner\n")
    
    # List available currency pairs in the repository
    print("3. Checking available data files...")
    available_pairs = []
    try:
        if os.path.exists(repo_path):
            for file in os.listdir(repo_path):
                if file.endswith('.csv'):
                    pair = file.split('_')[0]
                    available_pairs.append(pair)
            print(f"   ✓ Found {len(available_pairs)} potential pairs: {', '.join(available_pairs)}")
        else:
            print(f"   ! Repository path {repo_path} does not exist")
    except Exception as e:
        print(f"   ! Error listing files: {str(e)}")
    
    # Use first available pair or default to EURUSD
    test_pair = available_pairs[0] if available_pairs else 'EURUSD'
    print(f"   ✓ Using {test_pair} for testing\n")
    
    # 4. Define request
    print("4. Creating pipeline request...")
    request = {
        'pairs': [test_pair],
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',  # Using a wider date range
        'data_type': 'price',
        'align': True
    }
    print(f"   ✓ Request created for pair: {test_pair}\n")
    
    # 5. Create pipeline
    print("5. Creating pipeline...")
    try:
        pipeline = orchestrator.create_pipeline(request)
        print(f"   ✓ Pipeline created with {len(pipeline)} components")
        # Print pipeline components
        for i, component in enumerate(pipeline, 1):
            print(f"     {i}. {component.name} ({component.component_type})")
        print()
    except Exception as e:
        print(f"   ! Error creating pipeline: {str(e)}")
        return
    
    # 6. Execute pipeline
    print("6. Executing pipeline...")
    try:
        result = orchestrator.execute_pipeline(pipeline, request)
        print("   ✓ Pipeline executed successfully\n")
    except Exception as e:
        print(f"   ! Error executing pipeline: {str(e)}")
        return
    
    # 7. Analyze results
    print("7. Analyzing results...")
    if result:
        if isinstance(result, dict):
            if 'price' in result:
                # Structure with price dictionary
                for pair, df in result['price'].items():
                    print_dataframe_info(pair, df)
            elif all(isinstance(df, pd.DataFrame) for df in result.values()):
                # Dictionary of dataframes
                for pair, df in result.items():
                    print_dataframe_info(pair, df)
        elif isinstance(result, pd.DataFrame):
            # Single dataframe
            print_dataframe_info(test_pair, result)
        else:
            print(f"   ! Unknown result type: {type(result)}")
    else:
        print("   ! No results returned")

def print_dataframe_info(pair, df):
    """Helper function to print dataframe information"""
    if df is not None and isinstance(df, pd.DataFrame):
        print(f"   ✓ {pair} data processed successfully:")
        print(f"     - Shape: {df.shape} (rows, columns)")
        print(f"     - Date range: {df.index.min()} to {df.index.max()}")
        print(f"     - Columns: {', '.join(df.columns.tolist())}")
        
        # Calculate and display basic statistics
        if 'close' in df.columns:
            print(f"     - Price range: {df['close'].min():.4f} to {df['close'].max():.4f}")
        
        # Check for missing values
        missing = df.isna().sum().sum()
        if missing > 0:
            print(f"     - Warning: Contains {missing} missing values")
        print()
    else:
        print(f"   ! Invalid DataFrame for {pair}")

if __name__ == "__main__":
    test_full_pipeline()