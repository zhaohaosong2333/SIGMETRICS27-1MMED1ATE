"""Model transformer: COPSE → VESTA / IMM"""

from typing import List
from ..models.copse import COPSEModel
from ..models.vesta import VESTAModel
from ..models.imm import IMMModel
from ..utils.matrix_ops import mm_mul


def convert_copse_to_vesta(input_model: COPSEModel) -> VESTAModel:
    """Converts a COPSE model to a VESTA model."""
    current_model = VESTAModel()
    
    # Copy identical components
    current_model.thresh = input_model.thresh.copy()
    current_model.masks = [mask.copy() for mask in input_model.masks]
    
    # Copy parameters
    current_model.p = input_model.p  # Precision - critical for bit matrix generation
    current_model.q = input_model.q
    current_model.d = input_model.d
    current_model.b = input_model.b
    current_model.k = input_model.k
    current_model.name = input_model.name
    
    # Compute unified level matrix
    ulvls = []
    for lvl_i in range(input_model.d):
        ulvl = mm_mul(input_model.lvls[lvl_i].copy(), 
                      input_model.reshuf.copy())
        ulvls.append(ulvl)
    
    current_model.ulvls = ulvls
    return current_model


def convert_copse_to_imm(input_model: COPSEModel) -> IMMModel:
    """Converts a COPSE model to a 1MMED1ATE (IMM) model."""
    current_model = IMMModel()
    
    # Generate extended level matrix
    elvls = _generate_extended_lvls(
        input_model.lvls, 
        input_model.reshuf, 
        input_model.masks
    )
    
    # Generate extended matrix (EM)
    current_model.em = _generate_em(elvls)
    
    # Generate dual thresholds
    current_model.dthresh = _generate_dual_thresh(
        input_model.thresh, 
        input_model.p
    )
    
    # Copy parameters
    current_model.p = input_model.p
    current_model.q = input_model.q
    current_model.d = input_model.d
    current_model.b = input_model.b
    current_model.k = input_model.k
    current_model.name = input_model.name
    
    return current_model


def _generate_dual_thresh(thresh: List[int], p: int) -> List[int]:
    """Generates dual thresholds."""
    inv_thresh = [2**p - t for t in thresh]
    dual_thresh = []
    for i in range(len(thresh)):
        dual_thresh.extend([thresh[i], inv_thresh[i]])
    return dual_thresh


def _generate_extended_lvls(lvls, reshuf, masks):
    """Generates the extended level matrices (one w x 2q 0/1 array per level).

    Column 2j is the left edge (x < t_j) and 2j+1 the right edge (x >= t_j) of the branch
    that threshold slot j feeds; a leaf row selects the edge its path takes at that level.
    """
    import numpy as np

    reshuf_arr = np.asarray(reshuf, dtype=np.float64)
    elvls = []
    for lvl_i in range(len(lvls)):
        # unified level matrix lvl @ reshuf over GF(2), as in mm_mul (exact in float64)
        ulvl = np.rint(np.asarray(lvls[lvl_i], dtype=np.float64) @ reshuf_arr).astype(np.int64) % 2 != 0
        right = (np.asarray(masks[lvl_i]) != 0)[:, None]
        elvl = np.zeros((ulvl.shape[0], 2 * ulvl.shape[1]), dtype=np.int64)
        elvl[:, 0::2] = ulvl & ~right
        elvl[:, 1::2] = ulvl & right
        elvls.append(elvl)

    return elvls


def _generate_em(elvls) -> List[List[int]]:
    """Aggregates the extended matrix."""
    em = elvls[0].copy()
    for elvl in elvls[1:]:
        em += elvl
    return em.tolist()
