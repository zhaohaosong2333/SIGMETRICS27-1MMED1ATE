"""1MMED1ATE (IMM) model definition"""

from typing import List
from .base import BaseModel


class IMMModel(BaseModel):
    """1MMED1ATE model (edge-based single-matrix representation)."""
    
    def __init__(self):
        super().__init__()
        self.dthresh: List[int] = []  # Dual threshoulds
        self.em: List[List[int]] = [[]]  # Extended matrix
    
    def get_model_type(self) -> str:
        return "IMM"
