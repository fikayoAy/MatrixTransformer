# Time_Management/base.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class TimeSystemProtocol(Protocol):
    def is_initialized(self) -> bool:
        ...
    
    def get_current_time(self) -> float:
        ...


# Time_Management/base.py
from abc import ABC, abstractmethod
import numpy as np

class TimeSystemBase(ABC):
    """Base interface for time system functionality"""
    
    @abstractmethod
    def get_current_time(self) -> float:
        """Get current time value"""
        pass

    @abstractmethod
    def get_generator(self, seed) -> np.random.Generator:
        """Get random number generator"""
        pass