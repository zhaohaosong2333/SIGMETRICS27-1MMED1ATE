"""
FHE Scheme Abstract Base Class

Defines the interface for homomorphic encryption schemes (BGV, BFV, etc.)
"""

import math
from abc import ABC, abstractmethod
from typing import List, Any


class CtMatrix(list):
    """List of row ciphertexts that remembers how many slots each plaintext row filled."""
    slot_len = 0


class FHEScheme(ABC):
    """Abstract base class for FHE schemes"""

    def __init__(self):
        self.crypto_context = None
        self.key_pair = None
        # Replicate the MatMul input vector only as many times as the diagonal shape needs, and fail
        # loudly if the copies overflow one rotation row (N/2 slots) instead of silently wrapping
        # around. False restores the original always-3-copies behaviour, which returned wrong 1MM
        # results whenever 6q > N/2 (e.g. the MNIST RF at OD).
        self.exact_replication = True

    def _replication_rounds(self, ctxt_matrix: List[Any], colnum: int) -> int:
        """Rotate-and-add rounds the diagonal MatMul needs (copies - 1) for this operand."""
        rows = getattr(ctxt_matrix, "slot_len", 0)
        # mat_mul always builds at least two copies (v + rot(v, -colnum))
        copies = max(2, math.ceil((rows + colnum - 1) / colnum))
        row_slots = self.crypto_context.GetRingDimension() // 2
        if copies * colnum > row_slots:
            raise ValueError(f"MatMul needs {copies} x {colnum} = {copies * colnum} slots per rotation "
                             f"row but ring dimension {2 * row_slots} only has {row_slots}")
        return copies - 1

    @abstractmethod
    def generate_scheme(self, depth: int, rotation_list: List[int], plaintext_modulus: int = 65537):
        """
        Generate FHE scheme with specified parameters
        
        Args:
            depth: Multiplicative depth
            rotation_list: List of rotation amounts needed
            plaintext_modulus: Plaintext modulus (default: 65537)
        """
        pass
    
    @abstractmethod
    def encrypt_matrix(self, matrix: List[List[int]]) -> List[Any]:
        """
        Encrypt a matrix
        
        Args:
            matrix: 2D list of integers to encrypt
            
        Returns:
            List of ciphertexts (one per row)
        """
        pass
    
    @abstractmethod
    def encrypt_vector(self, vector: List[int]) -> Any:
        """
        Encrypt a vector
        
        Args:
            vector: List of integers to encrypt
            
        Returns:
            Ciphertext
        """
        pass
    
    @abstractmethod
    def decrypt(self, ciphertext: Any) -> Any:
        """
        Decrypt a ciphertext
        
        Args:
            ciphertext: Ciphertext to decrypt
            
        Returns:
            Decrypted plaintext
        """
        pass
    
    @abstractmethod
    def sec_comp_8bit(self, ctxt_a: List[Any], ctxt_b: List[Any], length: int) -> Any:
        """
        8-bit secure comparison (a < b)
        
        Args:
            ctxt_a: Encrypted bit matrix A (8 bits)
            ctxt_b: Encrypted bit matrix B (8 bits)
            length: Vector length
            
        Returns:
            Ciphertext of comparison result
        """
        pass
    
    @abstractmethod
    def sec_comp_16bit(self, ctxt_a: List[Any], ctxt_b: List[Any], length: int) -> Any:
        """
        16-bit secure comparison (a < b)
        
        Args:
            ctxt_a: Encrypted bit matrix A (16 bits)
            ctxt_b: Encrypted bit matrix B (16 bits)
            length: Vector length
            
        Returns:
            Ciphertext of comparison result
        """
        pass
    
    @abstractmethod
    def mat_mul(self, ctxt_matrix: List[Any], ctxt_vector: Any, 
                colnum: int, recursive_time: int = 2) -> Any:
        """
        Homomorphic matrix-vector multiplication
        
        Args:
            ctxt_matrix: Encrypted diagonalized matrix (list of ciphertexts)
            ctxt_vector: Encrypted vector
            colnum: Number of columns
            recursive_time: Recursion times for rotation
            
        Returns:
            Ciphertext of result vector
        """
        pass
    
    @abstractmethod
    def eval_add(self, ctxt_a: Any, ctxt_b: Any) -> Any:
        """
        Homomorphic addition
        
        Args:
            ctxt_a: First ciphertext
            ctxt_b: Second ciphertext
            
        Returns:
            Ciphertext of sum
        """
        pass
    
    @abstractmethod
    def eval_mult(self, ctxt_a: Any, ctxt_b: Any) -> Any:
        """
        Homomorphic multiplication
        
        Args:
            ctxt_a: First ciphertext
            ctxt_b: Second ciphertext
            
        Returns:
            Ciphertext of product
        """
        pass
    
    @abstractmethod
    def eval_sub(self, ctxt_a: Any, ctxt_b: Any) -> Any:
        """
        Homomorphic subtraction
        
        Args:
            ctxt_a: First ciphertext
            ctxt_b: Second ciphertext
            
        Returns:
            Ciphertext of difference
        """
        pass
    
    @abstractmethod
    def eval_negate(self, ctxt: Any) -> Any:
        """
        Homomorphic negation
        
        Args:
            ctxt: Ciphertext to negate
            
        Returns:
            Ciphertext of negated value
        """
        pass
    
    @abstractmethod
    def eval_rotate(self, ctxt: Any, amount: int) -> Any:
        """
        Homomorphic rotation
        
        Args:
            ctxt: Ciphertext to rotate
            amount: Rotation amount
            
        Returns:
            Rotated ciphertext
        """
        pass
