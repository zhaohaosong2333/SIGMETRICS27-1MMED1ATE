"""VESTA model definition"""

from typing import List
from .base import BaseModel


class VESTAModel(BaseModel):
    """VESTA model (unified level-based representation)."""
    
    def __init__(self):
        super().__init__()
        self.ulvls: List[List[List[int]]] = [[[]]]  # Unified level matrix
    
    def get_model_type(self) -> str:
        return "VESTA"
