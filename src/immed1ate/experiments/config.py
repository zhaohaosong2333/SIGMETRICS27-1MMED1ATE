"""
Experiment Configuration Module

Provides configuration classes and utilities for running experiments.
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run"""
    
    # Model parameters
    model_folder: str = ""
    model_files: List[str] = field(default_factory=list)
    precision: int = 0  # Model precision (8 or 16, 0 means auto-detect from file)
    
    # Protocol settings
    protocols: List[str] = field(default_factory=lambda: ["IMM", "VESTA", "COPSE"])
    bit_width: int = 8
    
    # FHE settings
    fhe_scheme: str = "BGV"  # BGV or BFV
    depth: int = -1  # Auto-calculate if -1
    config_setting: str = "OD"  # OD (OptDepth) or UD (UniDepth)
    
    # Experiment settings
    num_runs: int = 1  # Number of runs per model (for averaging)
    save_results: bool = True
    output_folder: str = "experiments/results"
    
    # Timing settings
    measure_preprocessing: bool = False
    measure_encryption: bool = True
    measure_evaluation: bool = True
    measure_decryption: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        valid_protocols = {"IMM", "VESTA", "COPSE"}
        for protocol in self.protocols:
            if protocol not in valid_protocols:
                raise ValueError(f"Invalid protocol: {protocol}. Must be one of {valid_protocols}")
        
        if self.fhe_scheme not in ["BGV", "BFV"]:
            raise ValueError(f"Invalid FHE scheme: {self.fhe_scheme}. Must be BGV or BFV")
        
        if self.bit_width not in [8, 16]:
            raise ValueError(f"Invalid bit width: {self.bit_width}. Must be 8 or 16")
        
        if self.config_setting not in ["OD", "UD"]:
            raise ValueError(f"Invalid config setting: {self.config_setting}. Must be OD or UD")


@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    
    # Model info
    model_name: str = ""
    model_params: Dict[str, int] = field(default_factory=dict)
    
    # Protocol results
    results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Timing breakdown (in milliseconds)
    preprocessing_time: float = 0.0
    encryption_time: float = 0.0
    evaluation_time: float = 0.0
    decryption_time: float = 0.0
    total_time: float = 0.0
    
    # Correctness
    results_match: bool = True
    predictions: Dict[str, Any] = field(default_factory=dict)
    
    def add_protocol_result(self, protocol: str, timing: Dict[str, float], prediction: Any = None):
        """Add timing results and prediction for a protocol"""
        self.results[protocol] = timing.copy()
        if prediction is not None:
            self.predictions[protocol] = prediction
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "model_name": self.model_name,
            "model_params": self.model_params,
            "results": self.results,
            "timing": {
                "preprocessing": self.preprocessing_time,
                "encryption": self.encryption_time,
                "evaluation": self.evaluation_time,
                "decryption": self.decryption_time,
                "total": self.total_time
            },
            "correctness": self.results_match
        }


def load_model_files(folder_path: str) -> List[str]:
    """
    Load all .mdl files from a folder
    
    Args:
        folder_path: Path to folder containing model files
        
    Returns:
        List of model file paths
    """
    path = Path(folder_path)
    if not path.exists():
        raise FileNotFoundError(f"Model folder not found: {folder_path}")
    
    model_files = sorted([str(f) for f in path.glob("*.mdl")])
    
    if not model_files:
        raise ValueError(f"No .mdl files found in: {folder_path}")
    
    return model_files


def calculate_fhe_depth(p: int, d: int, protocol: str, config_setting: str = "OD") -> int:
    """
    Calculate required FHE multiplicative depth
    
    Args:
        p: Bit width
        d: Tree depth
        protocol: Protocol name (IMM, VESTA, COPSE)
        config_setting: Configuration setting (OD or UD)
        
    Returns:
        Required multiplicative depth
    """
    if config_setting == "UD":
        # UD mode: all protocols use COPSE's depth calculation
        return p.bit_length() + d.bit_length() + 2
    else:
        # OD mode: each protocol uses its own depth calculation
        if protocol == "IMM":
            return p.bit_length() + 1
        elif protocol == "VESTA":
            return p.bit_length() + d.bit_length() + 1
        elif protocol == "COPSE":
            return p.bit_length() + d.bit_length() + 2
        else:
            raise ValueError(f"Unknown protocol: {protocol}")
