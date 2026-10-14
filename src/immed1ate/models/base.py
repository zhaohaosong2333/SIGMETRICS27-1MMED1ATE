"""Base model class."""

from abc import ABC, abstractmethod
from typing import List


class BaseModel(ABC):
    """Base model class."""
    
    def __init__(self):
        # Elements
        self.thresh: List[int] = []
        self.masks: List[List[int]] = [[]]
        
        # Parameters
        self.p: int = 8      # Precision
        self.q: int = -1     # Thresholds number
        self.d: int = -1     # Depths
        self.b: int = -1     # Branch number
        self.k: int = -1     # multiplicity
        
        # File name
        self.name: str = "undefined"
    
    @abstractmethod
    def get_model_type(self) -> str:
        """Return model type"""
        pass
