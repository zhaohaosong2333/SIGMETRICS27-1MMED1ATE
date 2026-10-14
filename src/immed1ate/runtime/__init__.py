"""
Runtime Module

Provides protocol implementations for different tree ensemble inference methods:
- IMM (1MMEDIATE): One matrix-vector multiplication (our innovation)
- VESTA: Unified level matrices (d multiplications)
- COPSE: Traditional node-based approach (baseline)
"""

from .protocol import BaseProtocol, RuntimeInput, CTRuntimeInput
from .imm_protocol import IMMProtocol, IMMRuntimeInput, CTIMMRuntimeInput
from .vesta_protocol import VESTAProtocol, VESTARuntimeInput, CTVESTARuntimeInput
from .copse_protocol import COPSEProtocol, CPSERuntimeInput, CTCPSERuntimeInput

__all__ = [
    # Abstract base classes
    'BaseProtocol',
    'RuntimeInput',
    'CTRuntimeInput',
    
    # IMM Protocol (1MMEDIATE)
    'IMMProtocol',
    'IMMRuntimeInput',
    'CTIMMRuntimeInput',
    
    # VESTA Protocol
    'VESTAProtocol',
    'VESTARuntimeInput',
    'CTVESTARuntimeInput',
    
    # COPSE Protocol
    'COPSEProtocol',
    'CPSERuntimeInput',
    'CTCPSERuntimeInput',
]
