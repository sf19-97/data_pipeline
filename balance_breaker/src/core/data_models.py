"""
Core Data Models

This module defines the core data models used across all subsystems.
These models provide a common language for the entire system.
"""

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Dict, List, Any, Optional, Union, TypeVar, Generic, Type
from enum import Enum
import datetime
import uuid
import json
import copy
import inspect


class ModelValidationError(Exception):
    """Exception raised for data model validation errors"""
    pass


class Direction(Enum):
    """Trade direction enum"""
    LONG = 1
    SHORT = -1


class AllocationAction(Enum):
    """Possible allocation actions"""
    CREATE = "create"         # Create a new position
    INCREASE = "increase"     # Increase an existing position
    DECREASE = "decrease"     # Decrease an existing position
    CLOSE = "close"           # Close an existing position
    REBALANCE = "rebalance"   # Rebalance an existing position


@dataclass
class MarketContext:
    """Current market conditions and instrument data"""
    price: float
    instrument: str
    timestamp: datetime.datetime
    pip_value: float
    pip_factor: float  # 100 for JPY, 10000 for others
    volatility: Optional[float] = None  # ATR or similar measure
    spread: Optional[float] = None
    regime: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class AccountState:
    """Current account state"""
    balance: float
    equity: float
    open_positions: Dict[str, Any]
    drawdown: float = 0.0
    high_water_mark: float = 0.0


@dataclass
class TradeParameters:
    """Complete trade parameters"""
    instrument: str
    direction: Union[Direction, int]
    entry_price: float
    stop_loss: float
    take_profit: List[float]  # Support multiple targets
    position_size: float
    risk_amount: float  # Absolute risk in account currency
    risk_percent: float  # Risk as percentage of account
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    additional_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and normalize data after initialization"""
        # Ensure direction is a Direction enum
        if isinstance(self.direction, int):
            self.direction = Direction.LONG if self.direction == 1 else Direction.SHORT
        
        # Ensure take_profit is a list
        if not isinstance(self.take_profit, list):
            self.take_profit = [self.take_profit]
        
        # Validate essential fields
        if self.entry_price <= 0:
            raise ModelValidationError(f"Invalid entry price: {self.entry_price}")
        
        if self.position_size <= 0:
            raise ModelValidationError(f"Invalid position size: {self.position_size}")
        
        if self.risk_percent < 0 or self.risk_percent > 1:
            raise ModelValidationError(f"Invalid risk percent: {self.risk_percent}")


@dataclass
class PortfolioPosition:
    """Portfolio position that extends risk management position with portfolio metadata"""
    instrument: str
    direction: Union[Direction, int]
    entry_price: float
    position_size: float
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stop_loss: Optional[float] = None
    take_profit: Optional[List[float]] = None
    entry_time: Optional[datetime.datetime] = None
    last_update_time: Optional[datetime.datetime] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    strategy_name: Optional[str] = None
    risk_amount: float = 0.0
    risk_percent: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize data after initialization"""
        # Ensure direction is an int (1 or -1)
        if isinstance(self.direction, Direction):
            self.direction = self.direction.value
        
        # Ensure take_profit is a list if provided
        if self.take_profit is not None and not isinstance(self.take_profit, list):
            self.take_profit = [self.take_profit]
        
        # Set default timestamps if not provided
        if self.entry_time is None:
            self.entry_time = datetime.datetime.now()
        
        if self.last_update_time is None:
            self.last_update_time = self.entry_time
        
        # Validate essential fields
        if self.entry_price <= 0:
            raise ModelValidationError(f"Invalid entry price: {self.entry_price}")
        
        if self.position_size <= 0:
            raise ModelValidationError(f"Invalid position size: {self.position_size}")


@dataclass
class AllocationInstruction:
    """Instruction for position allocation"""
    instrument: str
    action: AllocationAction
    direction: Union[Direction, int]
    target_size: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[List[float]] = None
    risk_percent: float = 0.0
    position_id: Optional[str] = None
    strategy_name: Optional[str] = None
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize data after initialization"""
        # Ensure direction is an int (1 or -1)
        if isinstance(self.direction, Direction):
            self.direction = self.direction.value
        
        # Ensure take_profit is a list if provided
        if self.take_profit is not None and not isinstance(self.take_profit, list):
            self.take_profit = [self.take_profit]
        
        # Validate essential fields
        if self.target_size < 0:
            raise ModelValidationError(f"Invalid target size: {self.target_size}")
        
        if self.risk_percent < 0 or self.risk_percent > 1:
            raise ModelValidationError(f"Invalid risk percent: {self.risk_percent}")


@dataclass
class Portfolio:
    """Portfolio data model representing the complete portfolio state"""
    name: str
    base_currency: str
    positions: Dict[str, PortfolioPosition] = field(default_factory=dict)
    initial_capital: float = 100000.0
    current_equity: float = 100000.0
    cash: float = 100000.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    high_water_mark: float = 100000.0
    creation_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    last_update_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    transaction_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and normalize data after initialization"""
        # Validate essential fields
        if self.initial_capital <= 0:
            raise ModelValidationError(f"Invalid initial capital: {self.initial_capital}")
        
        if self.current_equity <= 0:
            raise ModelValidationError(f"Invalid current equity: {self.current_equity}")
        
        if self.cash < 0:
            raise ModelValidationError(f"Invalid cash amount: {self.cash}")
        
        # Set high water mark to initial capital if not provided
        if self.high_water_mark <= 0:
            self.high_water_mark = self.initial_capital
    
    @property
    def drawdown(self) -> float:
        """Calculate current drawdown as percentage"""
        if self.high_water_mark <= 0:
            return 0.0
        return max(0.0, 1.0 - (self.current_equity / self.high_water_mark))
    
    @property
    def total_exposure(self) -> float:
        """Calculate total exposure as sum of all position risk percentages"""
        return sum(position.risk_percent for position in self.positions.values())
    
    @property
    def position_count(self) -> int:
        """Get current number of open positions"""
        return len(self.positions)
    
    def update_equity(self) -> float:
        """Update current equity based on position PnL"""
        self.unrealized_pnl = sum(position.unrealized_pnl for position in self.positions.values())
        self.current_equity = self.cash + self.unrealized_pnl
        
        # Update high water mark if needed
        if self.current_equity > self.high_water_mark:
            self.high_water_mark = self.current_equity
            
        self.last_update_time = datetime.datetime.now()
        return self.current_equity
    
    def add_transaction(self, transaction_type: str, details: Dict[str, Any]) -> None:
        """Add a transaction to the history"""
        transaction = {
            'timestamp': datetime.datetime.now(),
            'type': transaction_type,
            **details
        }
        self.transaction_history.append(transaction)
        
    def get_position_by_id(self, position_id: str) -> Optional[PortfolioPosition]:
        """Get position by ID"""
        for position in self.positions.values():
            if position.position_id == position_id:
                return position
        return None


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_profit_per_trade: float = 0.0
    avg_loss_per_trade: float = 0.0
    avg_trade: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    correlation: float = 0.0
    time_window: str = "all"  # 'day', 'week', 'month', 'year', 'all'
    start_date: Optional[datetime.datetime] = None
    end_date: Optional[datetime.datetime] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


# Helper functions for model manipulation

def clone_model(model: Any) -> Any:
    """
    Create a deep copy of a data model
    
    Args:
        model: Data model to clone
        
    Returns:
        Deep copy of the model
    """
    if is_dataclass(model):
        return copy.deepcopy(model)
    else:
        raise TypeError(f"Expected a dataclass, got {type(model)}")


def model_to_dict(model: Any) -> Dict[str, Any]:
    """
    Convert a data model to a dictionary
    
    Args:
        model: Data model to convert
        
    Returns:
        Dictionary representation of the model
    """
    if is_dataclass(model):
        return asdict(model)
    else:
        raise TypeError(f"Expected a dataclass, got {type(model)}")


def model_to_json(model: Any) -> str:
    """
    Convert a data model to a JSON string
    
    Args:
        model: Data model to convert
        
    Returns:
        JSON string representation of the model
    """
    if is_dataclass(model):
        def json_serializer(obj):
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, uuid.UUID):
                return str(obj)
            raise TypeError(f"Type {type(obj)} not serializable")
        
        return json.dumps(asdict(model), default=json_serializer)
    else:
        raise TypeError(f"Expected a dataclass, got {type(model)}")