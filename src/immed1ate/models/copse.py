"""COPSE model definition"""

from typing import List
from .base import BaseModel


class COPSEModel(BaseModel):
    """COPSE model (Node based)"""
    
    def __init__(self):
        super().__init__()
        self.lvls: List[List[List[int]]] = [[[]]]  # Level matrix
        self.reshuf: List[List[int]] = [[]]         # Reorder matrix
    
    def get_model_type(self) -> str:
        return "COPSE"
