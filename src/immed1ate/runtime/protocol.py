"""
Runtime Protocol Abstract Base Class

Defines the interface for all inference protocols (IMM, VESTA, COPSE)
"""

from abc import ABC, abstractmethod
from typing import List, Any
from ..models.base import BaseModel


class RuntimeInput(ABC):
    """Abstract base class for protocol inputs"""
    pass


class CTRuntimeInput(ABC):
    """Abstract base class for ciphertext protocol inputs"""
    pass


class BaseProtocol(ABC):
    """Abstract base class for inference protocols"""
    
    def __init__(self):
        self.fhe_scheme = None
    
    @abstractmethod
    def preprocessing(self, model: BaseModel, input_data: List[int]) -> RuntimeInput:
        """
        Preprocess model and input data
        
        Args:
            model: The tree ensemble model
            input_data: Input feature vector
            
        Returns:
            Preprocessed runtime input
        """
        pass
    
    @abstractmethod
    def generate_fhe_scheme(self, model: BaseModel, depth: int = -1) -> Any:
        """
        Generate FHE scheme with appropriate parameters
        
        Args:
            model: The tree ensemble model
            depth: Multiplicative depth (auto-calculated if -1)
            
        Returns:
            FHE scheme instance
        """
        pass
    
    @abstractmethod
    def encryption(self, fhe_scheme: Any, runtime_input: RuntimeInput) -> CTRuntimeInput:
        """
        Encrypt preprocessed data
        
        Args:
            fhe_scheme: FHE scheme instance
            runtime_input: Preprocessed runtime input
            
        Returns:
            Ciphertext runtime input
        """
        pass
    
    @abstractmethod
    def evaluate(self, fhe_scheme: Any, ct_runtime_input: CTRuntimeInput, 
                 bit_width: int = 8) -> Any:
        """
        Perform homomorphic evaluation
        
        Args:
            fhe_scheme: FHE scheme instance
            ct_runtime_input: Ciphertext runtime input
            bit_width: Bit width for secure comparison (8 or 16)
            
        Returns:
            Ciphertext result
        """
        pass
    
    @abstractmethod
    def decrypt(self, fhe_scheme: Any, ct_result: Any, model: BaseModel) -> List[bool]:
        """
        Decrypt and interpret results
        
        Args:
            fhe_scheme: FHE scheme instance
            ct_result: Ciphertext result
            model: The tree ensemble model
            
        Returns:
            Decrypted boolean results
        """
        pass
    
    def run_protocol(self, input_data: List[int], model: BaseModel, 
                     depth: int = -1) -> List[bool]:
        """
        Run complete inference protocol
        
        Args:
            input_data: Input feature vector
            model: The tree ensemble model
            depth: Multiplicative depth (auto-calculated if -1)
            
        Returns:
            Final prediction results
        """
        # Step 1: Preprocessing
        runtime_input = self.preprocessing(model, input_data)
        
        # Step 2: Generate FHE scheme
        fhe_scheme = self.generate_fhe_scheme(model, depth)
        
        # Step 3: Encryption
        ct_runtime_input = self.encryption(fhe_scheme, runtime_input)
        
        # Step 4: Evaluation
        ct_result = self.evaluate(fhe_scheme, ct_runtime_input, model.p)
        
        # Step 5: Decryption
        result = self.decrypt(fhe_scheme, ct_result, model)
        
        return result
