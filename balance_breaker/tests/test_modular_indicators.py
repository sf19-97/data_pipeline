"""
Test Script for Modular Indicators Architecture

Tests the updated orchestrator with modular indicators.
"""

import os
import pandas as pd

# Import orchestrator and core components
from balance_breaker.src.data_pipeline import (
    DataPipelineOrchestrator,
    PriceLoader, 
    DataNormalizer,
    DataValidator,
    TimeAligner
)

def test_modular_indicators():
    """Test the orchestrator with modular indicators"""
    print("Testing Balance Breaker Data Pipeline with Modular Indicators...\n")
    
    # 1. Create orchestrator
    print("1. Creating orchestrator...")
    orchestrator = DataPipelineOrchestrator()
    print("   ✓ Orchestrator created\n")
    
    # 2. Register core components
    print("2. Registering core components...")
    repo_path = os.path.join('balance_breaker', 'data', 'price')
    if not os.path.exists(repo_path):
        print(f"   ! Warning: Repository path {repo_path} does not exist")
        # Try alternative paths
        if os.path.exists('data/price'):
            repo_path = 'data/price'
            print(f"   ✓ Using alternative path: {repo_path}")
    
    # Register core components
    price_loader = PriceLoader({'repository_path': repo_path})
    orchestrator.register_component(price_loader)
    print(f"   ✓ Registered PriceLoader with repo path: {repo_path}")
    
    data_validator = DataValidator()
    orchestrator.register_component(data_validator)
    print("   ✓ Registered DataValidator")
    
    normalizer = DataNormalizer()
    orchestrator.register_component(normalizer)
    print("   ✓ Registered DataNormalizer")
    
    aligner = TimeAligner()
    orchestrator.register_component(aligner)
    print("   ✓ Registered TimeAligner\n")
    
    # 3. Register modular indicators
    print("3. Registering modular indicators...")
    indicator_count = orchestrator.register_modular_indicators()
    print(f"   ✓ Registered {indicator_count} modular indicators\n")
    
    # 4. List available currency pairs in the repository
    print("4. Checking available data files...")
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
    
    # 5. Create pipeline request with indicators
    print("5. Creating pipeline request with modular indicators...")
    request = {
        'pairs': [test_pair],
        'start_date': '2023-01-01',
        'end_date': '2023-03-31',  # Using a shorter period for faster execution
        'data_type': 'price',
        'align': True,
        'detect_gaps': True,
        'indicators': ['rsi', 'ma', 'macd', 'momentum', 'volatility']  # Request specific indicators
    }
    print(f"   ✓ Request created for pair: {test_pair} with indicators: {', '.join(request['indicators'])}\n")
    
    # 6. Create pipeline
    print("6. Creating pipeline...")
    try:
        pipeline = orchestrator.create_pipeline(request)
        print(f"   ✓ Pipeline created with {len(pipeline)} components")
        # Print pipeline components
        for i, component in enumerate(pipeline, 1):
            print(f"     {i}. {component.name} ({component.component_type})")
        print()
    except Exception as e:
        print(f"   ! Error creating pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. Execute pipeline
    print("7. Executing pipeline...")
    try:
        result = orchestrator.execute_pipeline(pipeline, request)
        print("   ✓ Pipeline executed successfully\n")
    except Exception as e:
        print(f"   ! Error executing pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 8. Analyze results
    print("8. Analyzing results...")
    if result:
        if isinstance(result, dict) and all(isinstance(df, pd.DataFrame) for df in result.values()):
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
        
        # Group columns by type
        grouped_columns = {}
        
        # Base columns
        base_cols = [col for col in df.columns if col in ['open', 'high', 'low', 'close', 'volume', 'pip_factor']]
        if base_cols:
            grouped_columns['Base'] = base_cols
        
        # RSI
        rsi_cols = [col for col in df.columns if 'RSI' in col]
        if rsi_cols:
            grouped_columns['RSI'] = rsi_cols
        
        # Moving Averages
        ma_cols = [col for col in df.columns if 'SMA_' in col or 'EMA_' in col]
        if ma_cols:
            grouped_columns['Moving Averages'] = ma_cols
        
        # MACD
        macd_cols = [col for col in df.columns if 'MACD' in col]
        if macd_cols:
            grouped_columns['MACD'] = macd_cols
        
        # Bollinger Bands
        bb_cols = [col for col in df.columns if 'BB_' in col]
        if bb_cols:
            grouped_columns['Bollinger Bands'] = bb_cols
        
        # ATR
        atr_cols = [col for col in df.columns if 'ATR' in col or 'TR' == col]
        if atr_cols:
            grouped_columns['ATR'] = atr_cols
        
        # Momentum
        momentum_cols = [col for col in df.columns if 'momentum_' in col or 'roc_' in col]
        if momentum_cols:
            grouped_columns['Momentum'] = momentum_cols
        
        # Volatility
        volatility_cols = [col for col in df.columns if 'volatility_' in col or 'hist_vol_' in col]
        if volatility_cols:
            grouped_columns['Volatility'] = volatility_cols
        
        # Other columns
        known_cols = sum([cols for cols in grouped_columns.values()], [])
        other_cols = [col for col in df.columns if col not in known_cols]
        if other_cols:
            grouped_columns['Other'] = other_cols
        
        # Print grouped columns
        for group, cols in grouped_columns.items():
            print(f"     - {group}: {len(cols)} columns")
            # Print first 5 columns in each group
            if cols:
                print(f"       {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}")
        
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
    test_modular_indicators()