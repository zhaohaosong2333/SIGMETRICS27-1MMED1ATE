"""
FHE (Fully Homomorphic Encryption) Module

Provides abstract interfaces and implementations for different FHE schemes.
"""

import sys
sys.path.append("/usr/local/lib/")
from openfhe import *

from .scheme import FHEScheme
from .bgv import BGVScheme
from .bfv import BFVScheme
from .operations import (
    fhe_reduction,
    fhe_add,
    fhe_sub,
    fhe_mult,
    fhe_negate,
    fhe_rotate
)

__all__ = [
    'FHEScheme',
    'BGVScheme', 
    'BFVScheme',
    'fhe_reduction',
    'fhe_add',
    'fhe_sub',
    'fhe_mult',
    'fhe_negate',
    'fhe_rotate'
]