"""
Compiler Module - Model extraction, transformation, and serialization

This module provides functions for:
- Extracting models from .mdl files
- Converting between model formats (COPSE ↔ VESTA ↔ IMM)
- Writing compiled models to files
"""

# Extractor functions
from .extractor import (
    create_node,
    read_forest_from_file,
    get_copse_model_from_serial_file,
    unsecure_copse_model,
    get_unsecure_copse_model_from_serial_model,
)

# Transformer functions
from .transformer import (
    convert_copse_to_vesta,
    convert_copse_to_imm,
)

# Writer functions
from .writer import (
    write_copse_model,
    write_vesta_model,
    write_imm_model,
)

__all__ = [
    # Extractor
    'create_node',
    'read_forest_from_file',
    'get_copse_model_from_serial_file',
    'unsecure_copse_model',
    'get_unsecure_copse_model_from_serial_model',
    
    # Transformer
    'convert_copse_to_vesta',
    'convert_copse_to_imm',
    
    # Writer
    'write_copse_model',
    'write_vesta_model',
    'write_imm_model',
]
