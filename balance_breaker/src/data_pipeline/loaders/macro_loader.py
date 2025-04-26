"""
Macro Loader - Loader for macroeconomic data

This component loads macroeconomic data from repositories.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.data_pipeline.base import BaseLoader

@implements("DataLoader")
class MacroLoader(BaseLoader):
    """
    Loader for macroeconomic data
    
    Parameters:
    -----------
    repository_path : str
        Path to the repository directory (default: None)
    derived_filename : str
        Name of the derived indicators file (default: 'derived_indicators.csv')
    file_prefix : str
        Prefix for macro data files (default: 'macro_')
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'repository_path': None,
            'derived_filename': 'derived_indicators.csv',
            'file_prefix': 'macro_'
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def load_data(self, context: Dict[str, Any]) -> pd.DataFrame:
        """Load macroeconomic data
        
        Args:
            context: Pipeline context with parameters:
                - start_date: Start date (YYYY-MM-DD)
                - end_date: End date (YYYY-MM-DD)
                - repository: Repository name (optional)
                
        Returns:
            DataFrame with macroeconomic indicators
        """
        try:
            start_date = context.get('start_date')
            end_date = context.get('end_date')
            repository = context.get('repository')
            
            # Determine repository path
            repo_path = self.parameters.get('repository_path')
            if repository:
                # Use repository from context if specified
                if 'repository_config' in context and repository in context['repository_config'].get('macro', {}):
                    repo_config = context['repository_config']['macro'][repository]
                    repo_path = repo_config.get('directory')
            
            if not repo_path:
                # Default to data/macro if no path specified
                repo_path = os.path.join('data', 'macro')
                self.logger.info(f"Using default repository path: {repo_path}")
            
            # Check if directory exists
            if not os.path.exists(repo_path):
                self.error_handler.handle_error(
                    ValueError(f"Repository directory not found: {repo_path}"),
                    context={'repo_path': repo_path},
                    subsystem='data_pipeline',
                    component='MacroLoader'
                )
                return pd.DataFrame()
            
            # Look for derived indicators first
            derived_filename = self.parameters.get('derived_filename', 'derived_indicators.csv')
            derived_path = os.path.join(repo_path, derived_filename)
            
            if os.path.exists(derived_path):
                self.logger.info(f"Loading derived indicators from {derived_path}")
                try:
                    macro_df = pd.read_csv(derived_path, index_col=0, parse_dates=True)
                    
                    # Apply date filtering
                    if start_date:
                        macro_df = macro_df[macro_df.index >= start_date]
                    if end_date:
                        macro_df = macro_df[macro_df.index <= end_date]
                    
                    return macro_df
                except Exception as e:
                    self.error_handler.handle_error(
                        e,
                        context={'file_path': derived_path},
                        subsystem='data_pipeline',
                        component='MacroLoader'
                    )
            
            # If derived indicators not found, load individual files
            self.logger.info("Derived indicators not found, loading individual files")
            all_data = {}
            file_prefix = self.parameters.get('file_prefix', 'macro_')
            
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if file.endswith('.csv') and file_prefix in file:
                        file_path = os.path.join(root, file)
                        try:
                            # Extract indicator name from filename
                            indicator = os.path.splitext(file)[0].split('_')[-1]
                            
                            # Load data
                            self.logger.debug(f"Loading macro data from {file_path}")
                            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                            
                            # If single column, use indicator as column name
                            if len(df.columns) == 1:
                                all_data[indicator] = df.iloc[:, 0]
                            else:
                                # Multiple columns, merge all
                                for col in df.columns:
                                    all_data[f"{indicator}_{col}"] = df[col]
                        except Exception as e:
                            self.error_handler.handle_error(
                                e,
                                context={'file': file},
                                subsystem='data_pipeline',
                                component='MacroLoader'
                            )
            
            # Combine all series into a DataFrame
            if all_data:
                macro_df = pd.DataFrame(all_data)
                
                # Fill NaN values
                self.logger.info("Filling NaN values in macro data")
                macro_df = macro_df.fillna(method='ffill').fillna(method='bfill')
                
                # Check for infinite values and replace with NaN, then fill
                self.logger.info("Checking for infinite values")
                macro_df = macro_df.replace([np.inf, -np.inf], np.nan)
                macro_df = macro_df.fillna(method='ffill').fillna(method='bfill')
                
                # Apply date filtering
                if start_date:
                    macro_df = macro_df[macro_df.index >= start_date]
                if end_date:
                    macro_df = macro_df[macro_df.index <= end_date]
                
                self.logger.info(f"Loaded macro data with {len(macro_df.columns)} indicators")
                return macro_df
            else:
                self.logger.warning(f"No valid macro data found in {repo_path}")
                return pd.DataFrame()
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={
                    'start_date': start_date if 'start_date' in locals() else None,
                    'end_date': end_date if 'end_date' in locals() else None
                },
                subsystem='data_pipeline',
                component='MacroLoader'
            )
            raise