"""Encoding/Decoding utilities for 1MMEDIATE"""

from typing import List
import random


def intvector2bitmatrix(vec: List[int], p: int = 8) -> List[List[int]]:
    """
    Convert integer vector to bit matrix (column-wise, OpenFHE format)
    
    This is the original implementation from Utils.py.
    Each row represents a bit position across all values.
    
    Args:
        vec: Input integer vector
        p: Number of bits per value (default: 8)
        
    Returns:
        Bit matrix where each row is a bit position
    """
    # Decompose first value
    current_bvec = _bit_decompose(vec[0], p)
    bmat = [[i] for i in current_bvec]
    
    # Process remaining values
    for i in range(1, len(vec)):
        current_bvec = _bit_decompose(vec[i], p)
        for j in range(p):
            bmat[j].append(current_bvec[j])
    
    return bmat


def _bit_decompose(value: int, bits: int) -> List[int]:
    """
    Decompose an integer into its bit representation
    
    Args:
        value: Integer value to decompose
        bits: Number of bits
        
    Returns:
        List of bits (LSB first)
    """
    return [(value >> i) & 1 for i in range(bits)]

def generate_random_vector(length: int, max_val: int = 1) -> List[int]:
    """Generates a random vector."""
    return [random.randint(0, max_val) for _ in range(length)]


def generate_random_matrix(rownum: int, colnum: int, max_val: int = 1) -> List[List[int]]:
    """Generates a random matrix."""
    return [[random.randint(0, max_val) for _ in range(colnum)] 
            for _ in range(rownum)]

def generate_selection_vector(target_element, source_vector: List) -> List[int]:
    """Generates a selection vector."""
    selection_vec = [0] * len(source_vector)
    target_index = source_vector.index(target_element)
    selection_vec[target_index] = 1
    return selection_vec


def generate_reorder_matrix(input_vec: List, output_vec: List) -> List[List[int]]:
    """Generates a reorder matrix."""
    reorder_mat = []
    for target_output in output_vec:
        reorder_mat.append(generate_selection_vector(target_output, input_vec))
    return reorder_mat
