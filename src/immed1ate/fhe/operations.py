"""
FHE Operations Module

Provides convenient wrapper functions for common FHE operations
"""

from typing import List, Any


def fhe_reduction(cc: Any, ctxt_matrix: List[Any]) -> Any:
    """
    Homomorphic reduction: multiply all elements in a list
    
    Uses a tree-based approach for efficiency.
    
    Args:
        cc: Crypto context
        ctxt_matrix: List of ciphertexts to reduce
        
    Returns:
        Single ciphertext (product of all inputs)
    """
    n = len(ctxt_matrix)
    op_lists = [2 ** i for i in range((n - 1).bit_length())][::-1]
    
    for op in op_lists:
        for i in range(op):
            if i + op >= n:
                break
            ctxt_matrix[i] = cc.eval_mult(ctxt_matrix[i], ctxt_matrix[i + op])
    
    return ctxt_matrix[0]


def fhe_add(cc: Any, ctxt_a: Any, ctxt_b: Any) -> Any:
    """Convenience wrapper for homomorphic addition"""
    return cc.EvalAdd(ctxt_a, ctxt_b)


def fhe_sub(cc: Any, ctxt_a: Any, ctxt_b: Any) -> Any:
    """Convenience wrapper for homomorphic subtraction"""
    return cc.EvalSub(ctxt_a, ctxt_b)


def fhe_mult(cc: Any, ctxt_a: Any, ctxt_b: Any) -> Any:
    """Convenience wrapper for homomorphic multiplication"""
    return cc.EvalMult(ctxt_a, ctxt_b)


def fhe_negate(cc: Any, ctxt: Any) -> Any:
    """Convenience wrapper for homomorphic negation"""
    return cc.EvalNegate(ctxt)


def fhe_rotate(cc: Any, ctxt: Any, amount: int) -> Any:
    """Convenience wrapper for homomorphic rotation"""
    return cc.EvalRotate(ctxt, amount)
