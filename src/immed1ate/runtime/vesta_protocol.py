"""
VESTA Protocol Implementation

Implements the VESTA inference protocol using unified level matrices.
Requires d matrix multiplications (one per depth level).
"""

from typing import List, Any
from .protocol import BaseProtocol, RuntimeInput, CTRuntimeInput
from ..models.vesta import VESTAModel
from ..fhe.scheme import FHEScheme
from ..fhe.operations import fhe_reduction
from ..utils.matrix_ops import diagonalize_matrix
from ..utils.encoding import intvector2bitmatrix


class VESTARuntimeInput(RuntimeInput):
    """Preprocessed input for VESTA protocol"""
    
    def __init__(self, facts: List[List[int]], thresh: List[List[int]], 
                 u_lvls: List[List[List[int]]], masks: List[List[int]]):
        """
        Initialize VESTA runtime input
        
        Args:
            facts: Facts bit matrix
            thresh: Thresholds bit matrix
            u_lvls: List of diagonalized unified level matrices
            masks: Level mask
        """
        self.facts = facts
        self.thresh = thresh
        self.u_lvls = u_lvls
        self.masks = masks


class CTVESTARuntimeInput(CTRuntimeInput):
    """Ciphertext input for VESTA protocol"""
    
    def __init__(self, ct_facts: Any, ct_thresh: Any, 
                 ct_u_lvls: List[Any], ct_masks: Any):
        """
        Initialize ciphertext VESTA runtime input
        
        Args:
            ct_facts: Encrypted facts
            ct_thresh: Encrypted thresholds
            ct_u_lvls: List of encrypted unified level matrices
            ct_masks: Encrypted masks
        """
        self.ct_facts = ct_facts
        self.ct_thresh = ct_thresh
        self.ct_u_lvls = ct_u_lvls
        self.ct_masks = ct_masks


class VESTAProtocol(BaseProtocol):
    """
    VESTA Protocol Implementation
    
    This protocol uses unified level matrices and requires d matrix-vector
    multiplications (one per depth level), plus a final reduction step.
    """
    
    def preprocessing(self, model: VESTAModel, 
                      input_data: List[int]) -> VESTARuntimeInput:
        """
        Preprocess model and input data for VESTA protocol
        
        Steps:
        1. Convert input to bit matrix
        2. Convert thresholds to bit matrix
        3. Diagonalize all unified level matrices
        
        Args:
            model: VESTA model
            input_data: Input feature vector
            
        Returns:
            Preprocessed VESTA runtime input
        """
        # Convert to bit matrices
        facts = intvector2bitmatrix(input_data, model.p)
        thresh = intvector2bitmatrix(model.thresh, model.p)
        
        # Diagonalize unified level matrices
        u_lvls = []
        for u_lvl in model.ulvls:
            u_lvls.append(diagonalize_matrix(u_lvl))
        
        return VESTARuntimeInput(facts, thresh, u_lvls, model.masks.copy())
    
    def generate_fhe_scheme(self, model: VESTAModel,
                            depth: int = -1,
                            scheme_type: str = "BGV") -> FHEScheme:
        """
        Generate FHE scheme for VESTA protocol
            
        For VESTA, the required depth is: log2(p) + log2(d) + 1
        where p is bit width and d is tree depth
            
        Args:
            model: VESTA model
            depth: Multiplicative depth (auto-calculated if -1)
            scheme_type: FHE scheme type ("BGV" or "BFV")
                
        Returns:
            FHE scheme instance (BGV or BFV)
        """
        from ..fhe.bgv import BGVScheme
        from ..fhe.bfv import BFVScheme
            
        if depth == -1:
            depth = model.p.bit_length() + model.d.bit_length() + 1
            
        # Rotation list for VESTA operations
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
                   runtime_input: VESTARuntimeInput) -> CTVESTARuntimeInput:
        """
        Encrypt preprocessed data for VESTA protocol
        
        Args:
            fhe_scheme: FHE scheme instance
            runtime_input: Preprocessed VESTA runtime input
            
        Returns:
            Ciphertext VESTA runtime input
        """
        ct_thresh = fhe_scheme.encrypt_matrix(runtime_input.thresh)
        ct_facts = fhe_scheme.encrypt_matrix(runtime_input.facts)
        
        # Encrypt all unified level matrices
        ct_u_lvls = []
        for u_lvl in runtime_input.u_lvls:
            ct_u_lvls.append(fhe_scheme.encrypt_matrix(u_lvl))
        
        ct_masks = fhe_scheme.encrypt_matrix(runtime_input.masks)
        
        return CTVESTARuntimeInput(ct_facts, ct_thresh, ct_u_lvls, ct_masks)
    
    def evaluate(self, fhe_scheme: FHEScheme, 
                 ct_runtime_input: CTVESTARuntimeInput,
                 bit_width: int = 8,
                 return_timing: bool = False) -> Any:
        """
        Perform homomorphic evaluation for VESTA protocol
        
        Steps:
        1. Secure comparison (single threshold comparison)
        2. Process each level (d matrix multiplications)
        3. Add masks at each level
        4. Reduce results across levels
        
        Args:
            fhe_scheme: FHE scheme instance
            ct_runtime_input: Ciphertext VESTA runtime input
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
                len(ct_runtime_input.ct_u_lvls[0])
            )
        else:
            ct_decisions = fhe_scheme.sec_comp_16bit(
                ct_runtime_input.ct_facts,
                ct_runtime_input.ct_thresh,
                len(ct_runtime_input.ct_u_lvls[0])
            )
        
        timer.timing()

        # Step 2: Process each level
        ct_lvl_results = []
        
        for i in range(len(ct_runtime_input.ct_u_lvls)):
            # Matrix multiplication for this level
            ct_current_lvl_decisions = fhe_scheme.mat_mul(
                ct_runtime_input.ct_u_lvls[i],
                ct_decisions,
                len(ct_runtime_input.ct_u_lvls[i])
            )
            
            # Add mask
            ct_level_result = fhe_scheme.eval_add(
                ct_current_lvl_decisions,
                ct_runtime_input.ct_masks[i]
            )
            
            ct_lvl_results.append(ct_level_result)
        
        timer.timing()
        
        # Step 3: Reduce across levels
        ct_leafs = fhe_reduction(fhe_scheme, ct_lvl_results)
        
        timer.timing()
        
        # Print detailed timing
        durations = timer.get_duration()
        timing_dict = {}
        if len(durations) >= 3:
            timing_dict = {
                'SecComp': durations[0],
                'LvlProcessing': durations[1],
                'Reduction': durations[2],
                'total_evaluation': sum(durations)
            }
            print(f"  VESTA-[SecComp,LvlProcessing,Reduction](ms): [{durations[0]:.2f}, {durations[1]:.2f}, {durations[2]:.2f}]")
        
        if return_timing:
            return ct_leafs, timing_dict
        return ct_leafs
    
    def decrypt(self, fhe_scheme: FHEScheme, ct_result: Any, 
                model: VESTAModel) -> List[bool]:
        """
        Decrypt and interpret results for VESTA protocol
        
        Args:
            fhe_scheme: FHE scheme instance
            ct_result: Ciphertext result
            model: VESTA model
            
        Returns:
            Decrypted boolean predictions
        """
        pt_leafs = fhe_scheme.decrypt(ct_result)
        
        # Set length and extract values
        pt_leafs.SetLength(len(model.ulvls[0]))
        leafs = pt_leafs.GetPackedValue()
        
        # Result: check which positions equal to 1
        result = [i == 1 for i in leafs]
        
        return result
