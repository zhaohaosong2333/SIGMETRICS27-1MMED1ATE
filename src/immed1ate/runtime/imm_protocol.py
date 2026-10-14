"""
1MMEDIATE (IMM) Protocol Implementation

Implements the 1MMEDIATE inference protocol using one homomorphic matrix-vector multiplication.
This is the core innovation of the project.
"""

from typing import List, Any
from .protocol import BaseProtocol, RuntimeInput, CTRuntimeInput
from ..models.imm import IMMModel
from ..fhe.scheme import FHEScheme
from ..utils.matrix_ops import diagonalize_matrix, plan_column_tiling
from ..utils.encoding import intvector2bitmatrix


class IMMRuntimeInput(RuntimeInput):
    """Preprocessed input for IMM protocol"""
    
    def __init__(self, d_facts: List[List[int]], d_thresh: List[List[int]], 
                 em: List[List[int]]):
        """
        Initialize IMM runtime input
        
        Args:
            d_facts: Dual facts bit matrix
            d_thresh: Dual thresholds bit matrix
            em: Extended level matrix
        """
        self.d_facts = d_facts
        self.d_thresh = d_thresh
        self.em = em
        self.tiles = 1          # > 1: each of the three fields is a list with one entry per column block


class CTIMMRuntimeInput(CTRuntimeInput):
    """Ciphertext input for IMM protocol"""
    
    def __init__(self, ct_d_facts: Any, ct_d_thresh: Any, ct_em: Any):
        """
        Initialize ciphertext IMM runtime input
        
        Args:
            ct_d_facts: Encrypted dual facts
            ct_d_thresh: Encrypted dual thresholds
            ct_em: Encrypted extended level matrix
        """
        self.ct_d_facts = ct_d_facts
        self.ct_d_thresh = ct_d_thresh
        self.ct_em = ct_em
        self.tiles = 1


class IMMProtocol(BaseProtocol):
    """
    1MMEDIATE (IMM) Protocol Implementation
    
    This protocol performs tree ensemble inference with only ONE homomorphic
    matrix-vector multiplication, making it highly efficient.

    Block tiling: the diagonal MatMul replicates its 2q-wide edge-decision vector inside one rotation
    row (N/2 slots). When that does not fit the ring the depth selects (very large forests, e.g. q > N/8
    at OD), set `row_slots = N/2` before preprocessing and the edge columns are split into K blocks of
    width c (utils.matrix_ops.plan_column_tiling): each block gets its own dual-feature / dual-threshold
    bit-planes, SecComp and diagonal MatMul, and the K partial leaf counts are summed. The number of
    ciphertext products stays 2q (plus padding to K*c), the multiplicative depth is unchanged, and with
    row_slots unset (or when one block fits) nothing changes.
    """

    def __init__(self):
        super().__init__()
        self.row_slots = None       # rotation-row capacity N/2 of the ring that will be used, if known
        self.tiles, self.tile_width = 1, None

    def _plan(self, model: IMMModel):
        rows, cols = len(model.em), len(model.em[0])
        plan = plan_column_tiling(rows, cols, self.row_slots) if self.row_slots else None
        self.tiles, self.tile_width = plan if plan and plan[0] > 1 else (1, None)

    def preprocessing(self, model: IMMModel, input_data: List[int]) -> IMMRuntimeInput:
        """
        Preprocess model and input data for IMM protocol
        
        Steps:
        1. Generate dual features from input
        2. Convert to bit matrices
        3. Diagonalize the extended matrix
        
        Args:
            model: IMM model
            input_data: Input feature vector
            
        Returns:
            Preprocessed IMM runtime input
        """
        # Generate dual features
        dual_x = self._generate_dual_features(input_data, model.p)
        self._plan(model)

        if self.tiles == 1:
            # Convert to bit matrices
            d_facts = intvector2bitmatrix(dual_x, model.p)
            d_thresh = intvector2bitmatrix(model.dthresh, model.p)

            # Diagonalize extended matrix
            em = diagonalize_matrix(model.em)

            return IMMRuntimeInput(d_facts, d_thresh, em)

        # Block-tiled layout: column block k covers edge columns [k*c, (k+1)*c), zero-padded to K*c
        # (a zero threshold never fires and a zero EM column adds nothing)
        c, cols = self.tile_width, len(model.em[0])
        pad = self.tiles * c - cols
        dx = (list(dual_x[:cols]) + [0] * max(0, cols - len(dual_x)) + [0] * pad)
        dt = list(model.dthresh[:cols]) + [0] * pad
        em_rows = [list(row) + [0] * pad for row in model.em]
        rt = IMMRuntimeInput(
            [intvector2bitmatrix(dx[k * c:(k + 1) * c], model.p) for k in range(self.tiles)],
            [intvector2bitmatrix(dt[k * c:(k + 1) * c], model.p) for k in range(self.tiles)],
            [diagonalize_matrix([row[k * c:(k + 1) * c] for row in em_rows]) for k in range(self.tiles)])
        rt.tiles = self.tiles
        return rt
    
    def generate_fhe_scheme(self, model: IMMModel, depth: int = -1,
                            scheme_type: str = "BGV") -> FHEScheme:
        """
        Generate FHE scheme for IMM protocol
        
        For IMM, the required depth is: log2(p) + 1
        where p is the bit width
        
        Args:
            model: IMM model
            depth: Multiplicative depth (auto-calculated if -1)
            scheme_type: FHE scheme type ("BGV" or "BFV")
            
        Returns:
            FHE scheme instance (BGV or BFV)
        """
        from ..fhe.bgv import BGVScheme
        from ..fhe.bfv import BFVScheme
        
        if depth == -1:
            depth = model.p.bit_length() + 1
        
        # Rotation list for IMM operations
        rotation_list = [-1, 1, -model.b * 2, -model.q * 2]
        if self.tiles > 1:
            rotation_list.append(-self.tile_width)
        
        # Select scheme based on scheme_type
        if scheme_type.upper() == "BGV":
            scheme = BGVScheme()
        elif scheme_type.upper() == "BFV":
            scheme = BFVScheme()
        else:
            raise ValueError(f"Unsupported FHE scheme: {scheme_type}. Use 'BGV' or 'BFV'.")
        
        scheme.generate_scheme(depth, rotation_list)
        
        return scheme
    
    def encryption(self, fhe_scheme: FHEScheme, 
                   runtime_input: IMMRuntimeInput) -> CTIMMRuntimeInput:
        """
        Encrypt preprocessed data for IMM protocol
        
        Args:
            fhe_scheme: FHE scheme instance
            runtime_input: Preprocessed IMM runtime input
            
        Returns:
            Ciphertext IMM runtime input
        """
        if getattr(runtime_input, "tiles", 1) > 1:
            ct = CTIMMRuntimeInput([fhe_scheme.encrypt_matrix(m) for m in runtime_input.d_facts],
                                   [fhe_scheme.encrypt_matrix(m) for m in runtime_input.d_thresh],
                                   [fhe_scheme.encrypt_matrix(m) for m in runtime_input.em])
            ct.tiles = runtime_input.tiles
            return ct

        ct_d_thresh = fhe_scheme.encrypt_matrix(runtime_input.d_thresh)
        ct_d_facts = fhe_scheme.encrypt_matrix(runtime_input.d_facts)
        ct_em = fhe_scheme.encrypt_matrix(runtime_input.em)
        
        return CTIMMRuntimeInput(ct_d_facts, ct_d_thresh, ct_em)
    
    def evaluate(self, fhe_scheme: FHEScheme, 
                 ct_runtime_input: CTIMMRuntimeInput,
                 bit_width: int = 8,
                 return_timing: bool = False) -> Any:
        """
        Perform homomorphic evaluation for IMM protocol
        
        This is the core of 1MMEDIATE:
        1. Secure comparison (dual threshold comparison)
        2. Single matrix-vector multiplication
        
        Args:
            fhe_scheme: FHE scheme instance
            ct_runtime_input: Ciphertext IMM runtime input
            bit_width: Bit width for secure comparison (8 or 16)
            return_timing: If True, returns (result, timing_dict)
            
        Returns:
            Ciphertext leaf predictions, or (predictions, timing_dict) if return_timing=True
        """
        from ..utils.helpers import Timer

        if getattr(ct_runtime_input, "tiles", 1) > 1:
            return self._evaluate_tiled(fhe_scheme, ct_runtime_input, bit_width, return_timing)

        timer = Timer()
        timer.timing()
        
        # Step 1: Secure comparison (dual threshold)
        if bit_width == 8:
            ct_d_decisions = fhe_scheme.sec_comp_8bit(
                ct_runtime_input.ct_d_facts,
                ct_runtime_input.ct_d_thresh,
                len(ct_runtime_input.ct_em)
            )
        else:
            ct_d_decisions = fhe_scheme.sec_comp_16bit(
                ct_runtime_input.ct_d_facts,
                ct_runtime_input.ct_d_thresh,
                len(ct_runtime_input.ct_em)
            )
        
        timer.timing()
        
        # Step 2: Matrix-vector multiplication (THE KEY INNOVATION!)
        ct_leafs = fhe_scheme.mat_mul(
            ct_runtime_input.ct_em,
            ct_d_decisions,
            len(ct_runtime_input.ct_em)
        )
        
        timer.timing()
        
        # Get detailed timing
        durations = timer.get_duration()
        timing_dict = {}
        if len(durations) >= 2:
            timing_dict = {
                'SecComp': durations[0],
                'MatMul': durations[1],
                'total_evaluation': sum(durations)
            }
            print(f"  IMM-[SecComp,MatMul](ms): [{durations[0]:.2f}, {durations[1]:.2f}]")
        
        if return_timing:
            return ct_leafs, timing_dict
        return ct_leafs
    
    def _evaluate_tiled(self, fhe_scheme, ct, bit_width, return_timing):
        """K column blocks: SecComp + diagonal MatMul per block, partial leaf counts summed."""
        import time
        sec = mm = 0.0
        ct_leafs = None
        for k in range(ct.tiles):
            width = len(ct.ct_em[k])
            t0 = time.time()
            if bit_width == 8:
                dec = fhe_scheme.sec_comp_8bit(ct.ct_d_facts[k], ct.ct_d_thresh[k], width)
            else:
                dec = fhe_scheme.sec_comp_16bit(ct.ct_d_facts[k], ct.ct_d_thresh[k], width)
            t1 = time.time()
            part = fhe_scheme.mat_mul(ct.ct_em[k], dec, width)
            ct_leafs = part if ct_leafs is None else fhe_scheme.eval_add(ct_leafs, part)
            t2 = time.time()
            sec += (t1 - t0) * 1000
            mm += (t2 - t1) * 1000
        timing_dict = {'SecComp': sec, 'MatMul': mm, 'total_evaluation': sec + mm, 'tiles': ct.tiles}
        print(f"  IMM-[SecComp,MatMul](ms) over {ct.tiles} column blocks: [{sec:.2f}, {mm:.2f}]")
        if return_timing:
            return ct_leafs, timing_dict
        return ct_leafs

    def decrypt(self, fhe_scheme: FHEScheme, ct_result: Any, 
                model: IMMModel) -> List[bool]:
        """
        Decrypt and interpret results for IMM protocol
        
        Args:
            fhe_scheme: FHE scheme instance
            ct_result: Ciphertext result
            model: IMM model
            
        Returns:
            Decrypted boolean predictions
        """
        pt_leafs = fhe_scheme.decrypt(ct_result)
        
        # Set length and extract values
        pt_leafs.SetLength(len(model.em))
        leafs = pt_leafs.GetPackedValue()
        
        # Result: check which positions equal to model depth
        result = [i == model.d for i in leafs]
        
        return result
    
    @staticmethod
    def _generate_dual_features(input_x: List[int], p: int) -> List[int]:
        """
        Generate dual features from input

        For each feature x, generate the dual pair (x, 2^p - 1 - x). The complement
        2^p-1-x pairs with the dual threshold 2^p-t so that the right-edge decision
        x >= t is computed as (2^p-1-x) < (2^p-t) by the same less-than circuit.
        (Using 2^p-x would instead compute x > t and mishandle exact ties x == t.)
        
        Args:
            input_x: Input feature vector
            p: Bit width
            
        Returns:
            Dual feature vector (twice the length)
        """
        dual_x = []
        for x in input_x:
            dual_x.append(x)
            dual_x.append((2 ** p) - 1 - x)
        return dual_x
