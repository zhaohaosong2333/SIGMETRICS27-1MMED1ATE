"""
1MMEDIATE: Private Tree Ensemble Inference via One Homomorphic Matrix-Vector Multiplication

A system for privacy-preserving tree ensemble inference using Fully Homomorphic Encryption (FHE).
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Export main modules
from .models.copse import COPSEModel
from .models.vesta import VESTAModel
from .models.imm import IMMModel

from .compiler.extractor import get_unsecure_copse_model_from_serial_model
from .compiler.transformer import convert_copse_to_vesta, convert_copse_to_imm
from .compiler.writer import write_copse_model, write_vesta_model, write_imm_model

from .utils.encoding import generate_random_vector, generate_random_matrix
from .utils.helpers import Timer, check_and_create_folder

from .experiments import ExperimentConfig, ExperimentResult, ExperimentRunner
