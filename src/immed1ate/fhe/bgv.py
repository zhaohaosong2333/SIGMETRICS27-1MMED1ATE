"""
BGV Scheme Implementation

Implements the BGV homomorphic encryption scheme using OpenFHE
"""

from typing import List, Any
from .scheme import FHEScheme, CtMatrix

class BGVScheme(FHEScheme):
    """BGV homomorphic encryption scheme implementation"""
    
    def __init__(self):
        super().__init__()
        try:
            from openfhe import (
                CCParamsBGVRNS, GenCryptoContext, PKESchemeFeature
            )
            self.openfhe_available = True
        except ImportError:
            self.openfhe_available = False
            print("Warning: OpenFHE not available. BGV scheme will not work.")
    
    @staticmethod
    def ring_dimension(depth: int, plaintext_modulus: int = 65537) -> int:
        """Ring dimension N OpenFHE picks for this depth at its default (128-bit) security."""
        from openfhe import CCParamsBGVRNS, GenCryptoContext
        parameters = CCParamsBGVRNS()
        parameters.SetPlaintextModulus(plaintext_modulus)
        parameters.SetMultiplicativeDepth(depth)
        return GenCryptoContext(parameters).GetRingDimension()

    def generate_scheme(self, depth: int, rotation_list: List[int], 
                       plaintext_modulus: int = 65537, ring_dim: int = None):
        """
        Generate BGV scheme with specified parameters
        
        Args:
            depth: Multiplicative depth
            rotation_list: List of rotation amounts needed
            plaintext_modulus: Plaintext modulus (default: 65537)
            ring_dim: Force a ring dimension (None = OpenFHE's choice for 128-bit security)
            
        Returns:
            self (for method chaining)
        """
        if not self.openfhe_available:
            raise ImportError("OpenFHE is required for BGV scheme")
        
        from openfhe import CCParamsBGVRNS, GenCryptoContext, PKESchemeFeature
        
        # Set parameters
        parameters = CCParamsBGVRNS()
        parameters.SetPlaintextModulus(plaintext_modulus)
        parameters.SetMultiplicativeDepth(depth)
        if ring_dim:
            parameters.SetRingDim(ring_dim)
        
        # Generate crypto context
        self.crypto_context = GenCryptoContext(parameters)
        
        # Enable required features
        self.crypto_context.Enable(PKESchemeFeature.PKE)
        self.crypto_context.Enable(PKESchemeFeature.KEYSWITCH)
        self.crypto_context.Enable(PKESchemeFeature.LEVELEDSHE)
        
        # Generate keys
        self.key_pair = self.crypto_context.KeyGen()
        
        # Generate evaluation keys
        self.crypto_context.EvalMultKeyGen(self.key_pair.secretKey)
        self.crypto_context.EvalRotateKeyGen(
            self.key_pair.secretKey, 
            rotation_list
        )
        
        return self
    
    def encrypt_matrix(self, matrix: List[List[int]]) -> List[Any]:
        """
        Encrypt a matrix (row by row)
        
        Args:
            matrix: 2D list of integers to encrypt
            
        Returns:
            List of ciphertexts (one per row)
        """
        ctxt_matrix = CtMatrix()
        ctxt_matrix.slot_len = max((len(vec) for vec in matrix), default=0)
        for vec in matrix:
            ptxt_vec = self.crypto_context.MakePackedPlaintext(vec)
            ctxt_vec = self.crypto_context.Encrypt(
                self.key_pair.publicKey, 
                ptxt_vec
            )
            ctxt_matrix.append(ctxt_vec)
        return ctxt_matrix
    
    def encrypt_vector(self, vector: List[int]) -> Any:
        """
        Encrypt a vector
        
        Args:
            vector: List of integers to encrypt
            
        Returns:
            Ciphertext
        """
        ptxt_vec = self.crypto_context.MakePackedPlaintext(vector)
        return self.crypto_context.Encrypt(
            self.key_pair.publicKey, 
            ptxt_vec
        )
    
    def decrypt(self, ciphertext: Any) -> Any:
        """
        Decrypt a ciphertext
        
        Args:
            ciphertext: Ciphertext to decrypt
            
        Returns:
            Decrypted plaintext object
        """
        return self.crypto_context.Decrypt(
            ciphertext, 
            self.key_pair.secretKey
        )
    
    def sec_comp_8bit(self, ctxt_a: List[Any], ctxt_b: List[Any], 
                     length: int) -> Any:
        """
        8-bit secure comparison (a < b)
        
        Implements the secure comparison protocol for 8-bit values.
        
        Args:
            ctxt_a: Encrypted bit matrix A (8 bits)
            ctxt_b: Encrypted bit matrix B (8 bits)
            length: Vector length
            
        Returns:
            Ciphertext of comparison result
        """
        # Initialize equality and less-than arrays
        eqr = [0]
        ltr = [self._fhe_lt(ctxt_a[0], ctxt_b[0], length)]
        
        # Compute equality and less-than for each bit
        for i in range(1, 8):
            eqr.append(self._fhe_eq(ctxt_a[i], ctxt_b[i], length))
            ltr.append(self._fhe_lt(ctxt_a[i], ctxt_b[i], length))
        
        # Build comparison result using tree structure
        e76 = self.crypto_context.EvalMult(eqr[7], eqr[6])
        e5l4 = self.crypto_context.EvalMult(eqr[5], ltr[4])
        e54 = self.crypto_context.EvalMult(eqr[5], eqr[4])
        e7654 = self.crypto_context.EvalMult(e76, e54)
        
        e3l2 = self.crypto_context.EvalMult(eqr[3], ltr[2])
        e32 = self.crypto_context.EvalMult(eqr[3], eqr[2])
        e32l1 = self.crypto_context.EvalMult(e32, ltr[1])
        e1l0 = self.crypto_context.EvalMult(eqr[1], ltr[0])
        e321l0 = self.crypto_context.EvalMult(e32, e1l0)
        
        ltr[6] = self.crypto_context.EvalMult(eqr[7], ltr[6])
        ltr[5] = self.crypto_context.EvalMult(e76, ltr[5])
        ltr[4] = self.crypto_context.EvalMult(e76, e5l4)
        ltr[3] = self.crypto_context.EvalMult(e7654, ltr[3])
        ltr[2] = self.crypto_context.EvalMult(e7654, e3l2)
        ltr[1] = self.crypto_context.EvalMult(e7654, e32l1)
        ltr[0] = self.crypto_context.EvalMult(e7654, e321l0)
        
        # Sum all less-than results
        for i in range(1, 8):
            ltr[0] = self.crypto_context.EvalAdd(ltr[0], ltr[i])
        
        return ltr[0]
    
    def sec_comp_16bit(self, ctxt_a: List[Any], ctxt_b: List[Any], 
                      length: int) -> Any:
        """
        16-bit secure comparison (a < b)
        
        Implements the secure comparison protocol for 16-bit values.
        
        Args:
            ctxt_a: Encrypted bit matrix A (16 bits)
            ctxt_b: Encrypted bit matrix B (16 bits)
            length: Vector length
            
        Returns:
            Ciphertext of comparison result
        """
        # Initialize equality and less-than arrays
        eqr = [0]
        ltr = [self._fhe_lt(ctxt_a[0], ctxt_b[0], length)]
        
        # Compute equality and less-than for each bit
        for i in range(1, 16):
            eqr.append(self._fhe_eq(ctxt_a[i], ctxt_b[i], length))
            ltr.append(self._fhe_lt(ctxt_a[i], ctxt_b[i], length))
        
        # Build comparison result using tree structure (16-bit version)
        e1514 = self.crypto_context.EvalMult(eqr[15], eqr[14])
        e13l12 = self.crypto_context.EvalMult(eqr[13], ltr[12])
        e1312 = self.crypto_context.EvalMult(eqr[13], eqr[12])
        e15141312 = self.crypto_context.EvalMult(e1514, e1312)
        
        e11l10 = self.crypto_context.EvalMult(eqr[11], ltr[10])
        e1110 = self.crypto_context.EvalMult(eqr[11], eqr[10])
        e1110l9 = self.crypto_context.EvalMult(e1110, ltr[9])
        e9l8 = self.crypto_context.EvalMult(eqr[9], ltr[8])
        e11109l8 = self.crypto_context.EvalMult(e1110, e9l8)
        e98 = self.crypto_context.EvalMult(eqr[9], eqr[8])
        e111098 = self.crypto_context.EvalMult(e1110, e98)
        e15141312111098 = self.crypto_context.EvalMult(e15141312, e111098)
        
        e7l6 = self.crypto_context.EvalMult(eqr[7], ltr[6])
        e76 = self.crypto_context.EvalMult(eqr[7], eqr[6])
        e76l5 = self.crypto_context.EvalMult(e76, ltr[5])
        e5l4 = self.crypto_context.EvalMult(eqr[5], ltr[4])
        e765l4 = self.crypto_context.EvalMult(e76, e5l4)
        e54 = self.crypto_context.EvalMult(eqr[5], eqr[4])
        e7654 = self.crypto_context.EvalMult(e76, e54)
        e7654l3 = self.crypto_context.EvalMult(e7654, ltr[3])
        e3l2 = self.crypto_context.EvalMult(eqr[3], ltr[2])
        e76543l2 = self.crypto_context.EvalMult(e7654, e3l2)
        e32 = self.crypto_context.EvalMult(eqr[3], eqr[2])
        e32l1 = self.crypto_context.EvalMult(e32, ltr[1])
        e765432l1 = self.crypto_context.EvalMult(e7654, e32l1)
        e1l0 = self.crypto_context.EvalMult(eqr[1], ltr[0])
        e321l0 = self.crypto_context.EvalMult(e32, e1l0)
        e7654321l0 = self.crypto_context.EvalMult(e7654, e321l0)
        
        ltr[14] = self.crypto_context.EvalMult(eqr[15], ltr[14])
        ltr[13] = self.crypto_context.EvalMult(e1514, ltr[13])
        ltr[12] = self.crypto_context.EvalMult(e1514, e13l12)
        ltr[11] = self.crypto_context.EvalMult(e15141312, ltr[11])
        ltr[10] = self.crypto_context.EvalMult(e15141312, e11l10)
        ltr[9] = self.crypto_context.EvalMult(e15141312, e1110l9)
        ltr[8] = self.crypto_context.EvalMult(e15141312, e11109l8)
        ltr[7] = self.crypto_context.EvalMult(e15141312111098, ltr[7])
        ltr[6] = self.crypto_context.EvalMult(e15141312111098, e7l6)
        ltr[5] = self.crypto_context.EvalMult(e15141312111098, e76l5)
        ltr[4] = self.crypto_context.EvalMult(e15141312111098, e765l4)
        ltr[3] = self.crypto_context.EvalMult(e15141312111098, e7654l3)
        ltr[2] = self.crypto_context.EvalMult(e15141312111098, e76543l2)
        ltr[1] = self.crypto_context.EvalMult(e15141312111098, e765432l1)
        ltr[0] = self.crypto_context.EvalMult(e15141312111098, e7654321l0)
        
        # Sum all less-than results
        for i in range(1, 16):
            ltr[0] = self.crypto_context.EvalAdd(ltr[0], ltr[i])
        
        return ltr[0]
    
    def mat_mul(self, ctxt_matrix: List[Any], ctxt_vector: Any,
                colnum: int, recursive_time: int = 2) -> Any:
        """
        Homomorphic matrix-vector multiplication
        
        Uses diagonalized matrix representation for efficient computation.
        
        Args:
            ctxt_matrix: Encrypted diagonalized matrix (list of ciphertexts)
            ctxt_vector: Encrypted vector
            colnum: Number of columns
            recursive_time: Recursion times for rotation
            
        Returns:
            Ciphertext of result vector
        """
        if self.exact_replication:
            recursive_time = self._replication_rounds(ctxt_matrix, colnum)
        # Initial rotation and addition
        ctxt_mv = self.crypto_context.EvalRotate(ctxt_vector, -colnum)
        ctxt_ev = self.crypto_context.EvalAdd(ctxt_mv, ctxt_vector)
        
        for i in range(recursive_time - 1):
            ctxt_mv = self.crypto_context.EvalRotate(ctxt_mv, -colnum)
            ctxt_ev = self.crypto_context.EvalAdd(ctxt_ev, ctxt_mv)
        
        # Matrix-vector multiplication
        rownum = len(ctxt_matrix)
        ctxt_r = self.crypto_context.EvalMult(ctxt_matrix[0], ctxt_ev)
        
        for i in range(1, rownum):
            ctxt_ev = self.crypto_context.EvalRotate(ctxt_ev, 1)
            ctxt_temp_r = self.crypto_context.EvalMult(ctxt_matrix[i], ctxt_ev)
            ctxt_r = self.crypto_context.EvalAdd(ctxt_temp_r, ctxt_r)
        
        return ctxt_r
    
    def eval_add(self, ctxt_a: Any, ctxt_b: Any) -> Any:
        """Homomorphic addition"""
        return self.crypto_context.EvalAdd(ctxt_a, ctxt_b)
    
    def eval_mult(self, ctxt_a: Any, ctxt_b: Any) -> Any:
        """Homomorphic multiplication"""
        return self.crypto_context.EvalMult(ctxt_a, ctxt_b)
    
    def eval_sub(self, ctxt_a: Any, ctxt_b: Any) -> Any:
        """Homomorphic subtraction"""
        return self.crypto_context.EvalSub(ctxt_a, ctxt_b)
    
    def eval_negate(self, ctxt: Any) -> Any:
        """Homomorphic negation"""
        return self.crypto_context.EvalNegate(ctxt)
    
    def eval_rotate(self, ctxt: Any, amount: int) -> Any:
        """Homomorphic rotation"""
        return self.crypto_context.EvalRotate(ctxt, amount)
    
    # Private helper methods
    
    def _fhe_eq(self, ctxt_a: Any, ctxt_b: Any, length: int) -> Any:
        """
        Homomorphic equality check (a == b)
        
        Computes: 1 - (a - b)^2
        """
        ptxt_ones = self.crypto_context.MakePackedPlaintext([1] * length)
        
        ctxt_sub = self.crypto_context.EvalSub(ctxt_b, ctxt_a)
        ctxt_square_sub = self.crypto_context.EvalMult(ctxt_sub, ctxt_sub)
        ctxt_nsquare_sub = self.crypto_context.EvalNegate(ctxt_square_sub)
        ctxt_eqr = self.crypto_context.EvalAdd(ctxt_nsquare_sub, ptxt_ones)
        
        return ctxt_eqr
    
    def _fhe_lt(self, ctxt_a: Any, ctxt_b: Any, length: int) -> Any:
        """
        Homomorphic less-than check for single bit (a < b)
        
        Computes: (1 - a) * b
        """
        ptxt_ones = self.crypto_context.MakePackedPlaintext([1] * length)
        
        ctxt_na = self.crypto_context.EvalNegate(ctxt_a)
        ctxt_not_a = self.crypto_context.EvalAdd(ctxt_na, ptxt_ones)
        ctxt_ltr = self.crypto_context.EvalMult(ctxt_not_a, ctxt_b)
        
        return ctxt_ltr
