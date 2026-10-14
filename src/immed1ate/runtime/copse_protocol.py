"""
COPSE Protocol Implementation

Implements the COPSE (node-based) inference protocol.
This is the traditional approach that requires multiple matrix multiplications.
"""

from typing import List, Any
from .protocol import BaseProtocol, RuntimeInput, CTRuntimeInput
from ..models.copse import COPSEModel
from ..fhe.scheme import FHEScheme
from ..fhe.operations import fhe_reduction
from ..utils.matrix_ops import diagonalize_matrix
from ..utils.encoding import intvector2bitmatrix


class CPSERuntimeInput(RuntimeInput):
    """Preprocessed input for COPSE protocol"""
    
    def __init__(self, facts: List[List[int]], thresh: List[List[int]], 
                 reshuf: List[List[int]], lvls: List[List[List[int]]], 
                 masks: List[List[int]]):
        """
        Initialize COPSE runtime input
        
        Args:
            facts: Facts bit matrix
            thresh: Thresholds bit matrix
            reshuf: Diagonalized reordering matrix
            lvls: List of diagonalized level matrices
            masks: Level masks
        """
        self.facts = facts
        self.thresh = thresh
        self.reshuf = reshuf
        self.lvls = lvls
        self.masks = masks


class CTCPSERuntimeInput(CTRuntimeInput):
    """Ciphertext input for COPSE protocol"""
    
    def __init__(self, ct_facts: Any, ct_thresh: Any, ct_reshuf: Any, 
                 ct_lvls: List[Any], ct_masks: Any):
        """
        Initialize ciphertext COPSE runtime input
        
        Args:
            ct_facts: Encrypted facts
            ct_thresh: Encrypted thresholds
            ct_reshuf: Encrypted reordering matrix
            ct_lvls: List of encrypted level matrices
            ct_masks: Encrypted masks
        """
        self.ct_facts = ct_facts
        self.ct_thresh = ct_thresh
        self.ct_reshuf = ct_reshuf
        self.ct_lvls = ct_lvls
        self.ct_masks = ct_masks


class COPSEProtocol(BaseProtocol):
    """
    COPSE Protocol Implementation
    
    This is the traditional node-based approach that requires:
    1. Secure comparison
    2. Reordering matrix multiplication
    3. Level-by-level processing (d matrix multiplications)
    4. Final reduction
    
    This is less efficient than 1MMEDIATE but serves as a baseline.
    """
    
    def preprocessing(self, model: COPSEModel, 
                      input_data: List[int]) -> CPSERuntimeInput:
        """
        Preprocess model and input data for COPSE protocol
        
        Steps:
        1. Convert input to bit matrix
        2. Convert thresholds to bit matrix
        3. Diagonalize reordering matrix
        4. Diagonalize all level matrices
        
        Args:
            model: COPSE model
            input_data: Input feature vector
            
        Returns:
            Preprocessed COPSE runtime input
        """
        # Convert to bit matrices
        facts = intvector2bitmatrix(input_data, model.p)
        thresh = intvector2bitmatrix(model.thresh, model.p)
        
        # Diagonalize reordering matrix
        reshuf = diagonalize_matrix(model.reshuf)
        
        # Diagonalize level matrices
        lvls = []
        for lvl in model.lvls:
            lvls.append(diagonalize_matrix(lvl))
        
        return CPSERuntimeInput(facts, thresh, reshuf, lvls, model.masks.copy())
    
    def generate_fhe_scheme(self, model: COPSEModel, 
                            depth: int = -1,
                            scheme_type: str = "BGV") -> FHEScheme:
        """
        Generate FHE scheme for COPSE protocol
        
        For COPSE, the required depth is: log2(p) + log2(d) + 2
        where p is bit width and d is tree depth
        
        Args:
            model: COPSE model
            depth: Multiplicative depth (auto-calculated if -1)
            scheme_type: FHE scheme type ("BGV" or "BFV")
            
        Returns:
            FHE scheme instance (BGV or BFV)
        """
        from ..fhe.bgv import BGVScheme
        from ..fhe.bfv import BFVScheme
        
        if depth == -1:
            depth = model.p.bit_length() + model.d.bit_length() + 2
        
        # Rotation list for COPSE operations
        rotation_list = [-1, 1, -model.b, -model.q]
        
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
                   runtime_input: CPSERuntimeInput) -> CTCPSERuntimeInput:
        """
        Encrypt preprocessed data for COPSE protocol
        
        Args:
            fhe_scheme: FHE scheme instance
            runtime_input: Preprocessed COPSE runtime input
            
        Returns:
            Ciphertext COPSE runtime input
        """
        ct_thresh = fhe_scheme.encrypt_matrix(runtime_input.thresh)
        ct_facts = fhe_scheme.encrypt_matrix(runtime_input.facts)
        
        # Encrypt reordering matrix
        ct_reshuf = fhe_scheme.encrypt_matrix(runtime_input.reshuf)
        
        # Encrypt all level matrices
        ct_lvls = []
        for lvl in runtime_input.lvls:
            ct_lvls.append(fhe_scheme.encrypt_matrix(lvl))
        
        ct_masks = fhe_scheme.encrypt_matrix(runtime_input.masks)
        
        return CTCPSERuntimeInput(ct_facts, ct_thresh, ct_reshuf, ct_lvls, ct_masks)
    
    def evaluate(self, fhe_scheme: FHEScheme, 
                 ct_runtime_input: CTCPSERuntimeInput,
                 bit_width: int = 8,
                 return_timing: bool = False) -> Any:
        """
        Perform homomorphic evaluation for COPSE protocol
        
        Steps:
        1. Secure comparison (single threshold comparison)
        2. Reordering matrix multiplication
        3. Process each level (d matrix multiplications)
        4. Add masks at each level
        5. Reduce results across levels
        
        Args:
            fhe_scheme: FHE scheme instance
            ct_runtime_input: Ciphertext COPSE runtime input
            bit_width: Bit width for secure comparison (8 or 16)
            return_timing: If True, returns (result, timing_dict)
            
        Returns:
            Ciphertext leaf predictions, or (predictions, timing_dict) if return_timing=True
        """
        from ..utils.helpers import Timer
        
        timer = Timer()
        timer.timing()
        
        # Step 1: Secure comparison
        if bit_width == 8:
            ct_decisions = fhe_scheme.sec_comp_8bit(
                ct_runtime_input.ct_facts,
                ct_runtime_input.ct_thresh,
                len(ct_runtime_input.ct_lvls[0])
            )
        else:
            ct_decisions = fhe_scheme.sec_comp_16bit(
                ct_runtime_input.ct_facts,
                ct_runtime_input.ct_thresh,
                len(ct_runtime_input.ct_lvls[0])
            )
        
        timer.timing()
        
        # Step 2: Reordering matrix multiplication
        ct_branches = fhe_scheme.mat_mul(
            ct_runtime_input.ct_reshuf,
            ct_decisions,
            len(ct_runtime_input.ct_reshuf)
        )
        
        timer.timing()
        
        # Step 3: Process each level
        ct_lvl_results = []
        
        for i in range(len(ct_runtime_input.ct_lvls)):
            # Matrix multiplication for this level
            ct_current_lvl_decisions = fhe_scheme.mat_mul(
                ct_runtime_input.ct_lvls[i],
                ct_branches,
                len(ct_runtime_input.ct_lvls[i])
            )
            
            # Add mask
            ct_level_result = fhe_scheme.eval_add(
                ct_current_lvl_decisions,
                ct_runtime_input.ct_masks[i]
            )
            
            ct_lvl_results.append(ct_level_result)
        
        timer.timing()
        
        # Step 4: Reduce across levels
        ct_leafs = fhe_reduction(fhe_scheme, ct_lvl_results)
        
        timer.timing()
        
        # Print detailed timing
        durations = timer.get_duration()
        timing_dict = {}
        if len(durations) >= 4:
            timing_dict = {
                'SecComp': durations[0],
                'Reordering': durations[1],
                'LvlProcessing': durations[2],
                'Reduction': durations[3],
                'total_evaluation': sum(durations)
            }
            print(f"  COPSE-[SecComp,Reordering,LvlProcessing,Reduction](ms): [{durations[0]:.2f}, {durations[1]:.2f}, {durations[2]:.2f}, {durations[3]:.2f}]")
        
        if return_timing:
            return ct_leafs, timing_dict
        return ct_leafs
    
    def decrypt(self, fhe_scheme: FHEScheme, ct_result: Any, 
                model: COPSEModel) -> List[bool]:
        """
        Decrypt and interpret results for COPSE protocol
        
        Args:
            fhe_scheme: FHE scheme instance
            ct_result: Ciphertext result
            model: COPSE model
            
        Returns:
            Decrypted boolean predictions
        """
        pt_leafs = fhe_scheme.decrypt(ct_result)
        
        # Set length and extract values
        pt_leafs.SetLength(len(model.lvls[0]))
        leafs = pt_leafs.GetPackedValue()
        
        # Result: check which positions equal to 1
        result = [i == 1 for i in leafs]
        
        return result