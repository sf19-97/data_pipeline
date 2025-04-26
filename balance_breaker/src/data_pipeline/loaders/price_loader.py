"""
Price Loader - Loader for price data from repositories

This component loads price data from file repositories.
"""

import os
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

from balance_breaker.src.core.interface_registry import implements
from balance_breaker.src.data_pipeline.base import BaseLoader

@implements("DataLoader")
class PriceLoader(BaseLoader):
    """
    Loader for price data from repositories
    
    Parameters:
    -----------
    repository_path : str
        Path to the repository directory (default: None)
    file_extensions : List[str]
        List of file extensions to search for (default: ['.csv', '.CSV', '.txt', '.TXT'])
    """
    
    def __init__(self, parameters=None):
        # Define default parameters
        default_params = {
            'repository_path': None,
            'file_extensions': ['.csv', '.CSV', '.txt', '.TXT']
        }
        
        # Initialize with parameters
        super().__init__(parameters or default_params)
    
    def load_data(self, context: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Load price data from repository
        
        Args:
            context: Pipeline context with parameters:
                - pairs: List of currency pairs
                - start_date: Start date (YYYY-MM-DD)
                - end_date: End date (YYYY-MM-DD)
                - repository: Repository name (optional)
                
        Returns:
            Dictionary of DataFrames with pair as key
        """
        try:
            pairs = context.get('pairs', [])
            start_date = context.get('start_date')
            end_date = context.get('end_date')
            repository = context.get('repository')
            
            if not pairs:
                self.logger.warning("No pairs specified for loading")
                return {}
            
            # Determine repository path
            repo_path = self.parameters.get('repository_path')
            if repository:
                # Use repository from context if specified
                if 'repository_config' in context and repository in context['repository_config'].get('price', {}):
                    repo_config = context['repository_config']['price'][repository]
                    repo_path = repo_config.get('directory')
            
            if not repo_path:
                # Default to data/price if no path specified
                repo_path = os.path.join('data', 'price')
                self.logger.info(f"Using default repository path: {repo_path}")
            
            # Check if directory exists
            if not os.path.exists(repo_path):
                error_msg = f"Repository directory not found: {repo_path}"
                self.error_handler.handle_error(
                    ValueError(error_msg),
                    context={'repo_path': repo_path},
                    subsystem='data_pipeline',
                    component='PriceLoader'
                )
                return {}
            
            # Load each pair
            data = {}
            for pair in pairs:
                try:
                    # Find file for pair
                    file_path = self._find_pair_file(repo_path, pair)
                    if not file_path:
                        self.logger.warning(f"No file found for pair {pair} in {repo_path}")
                        continue
                    
                    # Load the data
                    self.logger.info(f"Loading price data for {pair} from {file_path}")
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                    # Apply date filtering
                    if start_date:
                        df = df[df.index >= start_date]
                    if end_date:
                        df = df[df.index <= end_date]
                    
                    # Store in result dictionary
                    data[pair] = df
                    self.logger.info(f"Loaded {len(df)} rows for {pair}")
                    
                except Exception as e:
                    self.error_handler.handle_error(
                        e,
                        context={'pair': pair, 'file_path': file_path if 'file_path' in locals() else None},
                        subsystem='data_pipeline',
                        component='PriceLoader'
                    )
            
            return data
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context={'pairs': pairs if 'pairs' in locals() else []},
                subsystem='data_pipeline',
                component='PriceLoader'
            )
            raise
    
    def _find_pair_file(self, directory: str, pair: str) -> Optional[str]:
        """Find data file for the specified pair
        
        Args:
            directory: Repository directory
            pair: Currency pair to find
            
        Returns:
            Path to data file or None if not found
        """
        # Get extensions from parameters
        extensions = self.parameters.get('file_extensions', ['.csv', '.CSV', '.txt', '.TXT'])
        
        if not os.path.exists(directory):
            self.logger.error(f"Repository directory not found: {directory}")
            return None
        
        # Track all candidate files
        candidates = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Check if filename contains pair name and has correct extension
                if pair.lower() in file.lower() and any(file.lower().endswith(ext.lower()) for ext in extensions):
                    candidates.append(file_path)
                    
                    # Exact match prioritization
                    if file.lower() == f"{pair.lower()}.csv" or file.lower() == f"{pair.lower()}_h1.csv":
                        self.logger.debug(f"Found exact match for {pair}: {file}")
                        return file_path
        
        # Return first candidate if any found
        if candidates:
            self.logger.debug(f"Found candidate match for {pair}: {os.path.basename(candidates[0])}")
            return candidates[0]
        
        # No candidates found
        return None