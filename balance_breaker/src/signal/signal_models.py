"""
Signal Data Models

This module defines the core data models for the signal subsystem.
These models provide a common language for signals across the system.
"""

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Dict, List, Any, Optional, Union, Set, Type
from enum import Enum
import datetime
import uuid
import json
import copy

from balance_breaker.src.core.data_models import ModelValidationError


class SignalType(Enum):
    """Signal type enum to categorize signals"""
    RAW = "raw"                  # Raw observation without implied action
    INTERPRETED = "interpreted"  # Market meaning but no specific action
    ACTION = "action"            # Specific action signal


class SignalDirection(Enum):
    """Signal direction enum for directional signals"""
    BULLISH = 1     # Upward/positive direction
    NEUTRAL = 0     # No clear direction
    BEARISH = -1    # Downward/negative direction


class SignalStrength(Enum):
    """Signal strength enum"""
    WEAK = 1       # Weak signal
    MODERATE = 2   # Moderate signal
    STRONG = 3     # Strong signal
    VERY_STRONG = 4 # Very strong signal


class SignalPriority(Enum):
    """Signal priority enum"""
    LOW = 1        # Low priority, informational
    MEDIUM = 2     # Medium priority, should be considered
    HIGH = 3       # High priority, important
    CRITICAL = 4   # Critical priority, must be acted upon


class SignalConfidence(Enum):
    """Signal confidence enum"""
    SPECULATIVE = 1  # Speculative, low confidence
    PROBABLE = 2     # Probable, moderate confidence
    LIKELY = 3       # Likely, high confidence
    CONFIRMED = 4    # Confirmed, very high confidence


class Timeframe(Enum):
    """Timeframe enum for signals"""
    M1 = "1m"     # 1 minute
    M5 = "5m"     # 5 minutes
    M15 = "15m"   # 15 minutes
    M30 = "30m"   # 30 minutes
    H1 = "1h"     # 1 hour
    H4 = "4h"     # 4 hours
    D1 = "1d"     # 1 day
    W1 = "1w"     # 1 week
    MN1 = "1M"    # 1 month


@dataclass
class SignalMetadata:
    """Metadata for a signal"""
    source: str                                  # Signal source/generator
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    targets: List[str] = field(default_factory=list)  # Target subsystems
    lifespan: Optional[datetime.timedelta] = None     # Signal validity period
    confidence: SignalConfidence = SignalConfidence.PROBABLE  # Confidence level
    priority: SignalPriority = SignalPriority.MEDIUM          # Priority level
    tags: List[str] = field(default_factory=list)             # Signal tags
    context: Dict[str, Any] = field(default_factory=dict)     # Generation context


@dataclass
class Signal:
    """Base signal data model"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str                         # Trading symbol (e.g., 'EURUSD')
    timeframe: Timeframe                # Signal timeframe
    signal_type: SignalType             # Type of signal
    direction: SignalDirection          # Signal direction
    strength: SignalStrength            # Signal strength
    metadata: SignalMetadata            # Signal metadata
    creation_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    expiration_time: Optional[datetime.datetime] = None  # When signal expires
    parent_signals: List[str] = field(default_factory=list)  # IDs of parent signals
    child_signals: List[str] = field(default_factory=list)   # IDs of child signals
    data: Dict[str, Any] = field(default_factory=dict)       # Signal-specific data

    def __post_init__(self):
        """Validate and normalize data after initialization"""
        # Set expiration time based on lifespan if provided
        if self.metadata.lifespan and not self.expiration_time:
            self.expiration_time = self.metadata.timestamp + self.metadata.lifespan

        # Validate essential fields
        if not self.symbol:
            raise ModelValidationError(f"Invalid symbol: {self.symbol}")

    @property
    def is_expired(self) -> bool:
        """Check if signal is expired"""
        if not self.expiration_time:
            return False
        return datetime.datetime.now() > self.expiration_time

    @property
    def age(self) -> datetime.timedelta:
        """Get signal age"""
        return datetime.datetime.now() - self.metadata.timestamp


@dataclass
class RawSignal(Signal):
    """Raw observation signal without implied action"""
    
    def __post_init__(self):
        """Validate and normalize data after initialization"""
        # Ensure signal type is RAW
        self.signal_type = SignalType.RAW
        super().__post_init__()


@dataclass
class InterpretedSignal(Signal):
    """Interpreted signal with market meaning but no specific action"""
    interpretation: str = ""           # Human-readable interpretation
    indicators: Dict[str, Any] = field(default_factory=dict)  # Supporting indicators
    
    def __post_init__(self):
        """Validate and normalize data after initialization"""
        # Ensure signal type is INTERPRETED
        self.signal_type = SignalType.INTERPRETED
        super().__post_init__()


@dataclass
class ActionSignal(Signal):
    """Action signal with specific actions for subsystems"""
    action_type: str = ""              # Type of action to take
    action_parameters: Dict[str, Any] = field(default_factory=dict)  # Action parameters
    urgency: int = 1                   # Urgency level (1-5)
    
    def __post_init__(self):
        """Validate and normalize data after initialization"""
        # Ensure signal type is ACTION
        self.signal_type = SignalType.ACTION
        super().__post_init__()


@dataclass
class SignalGroup:
    """Group of related signals"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    signals: List[Signal] = field(default_factory=list)
    creation_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Helper functions for signal manipulation

def clone_signal(signal: Signal) -> Signal:
    """
    Create a deep copy of a signal
    
    Args:
        signal: Signal to clone
        
    Returns:
        Deep copy of the signal
    """
    if is_dataclass(signal):
        return copy.deepcopy(signal)
    else:
        raise TypeError(f"Expected a dataclass, got {type(signal)}")


def signal_to_dict(signal: Signal) -> Dict[str, Any]:
    """
    Convert a signal to a dictionary
    
    Args:
        signal: Signal to convert
        
    Returns:
        Dictionary representation of the signal
    """
    if is_dataclass(signal):
        return asdict(signal)
    else:
        raise TypeError(f"Expected a dataclass, got {type(signal)}")


def signal_to_json(signal: Signal) -> str:
    """
    Convert a signal to a JSON string
    
    Args:
        signal: Signal to convert
        
    Returns:
        JSON string representation of the signal
    """
    if is_dataclass(signal):
        def json_serializer(obj):
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, uuid.UUID):
                return str(obj)
            if is_dataclass(obj):
                return asdict(obj)
            raise TypeError(f"Type {type(obj)} not serializable")
        
        return json.dumps(asdict(signal), default=json_serializer)
    else:
        raise TypeError(f"Expected a dataclass, got {type(signal)}")