"""
Experiment Runner Module

Executes experiments and collects results.
"""

import json
import time
from typing import List, Dict, Any
from pathlib import Path

from .config import ExperimentConfig, ExperimentResult, load_model_files, calculate_fhe_depth
from ..models.copse import COPSEModel
from ..compiler.extractor import get_unsecure_copse_model_from_serial_model as extract_copse_model
from ..compiler.transformer import convert_copse_to_vesta, convert_copse_to_imm
from ..runtime.imm_protocol import IMMProtocol
from ..runtime.vesta_protocol import VESTAProtocol
from ..runtime.copse_protocol import COPSEProtocol
from ..utils.encoding import generate_random_vector


class ExperimentRunner:
    """
    Runs experiments comparing different protocols
    
    Supports:
    - Multiple models
    - Multiple protocols
    - Timing measurements
    - Correctness verification
    - Result serialization
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results: List[ExperimentResult] = []
        
        # Initialize protocols
        self.protocols = {}
        if "IMM" in config.protocols:
            self.protocols["IMM"] = IMMProtocol()
        if "VESTA" in config.protocols:
            self.protocols["VESTA"] = VESTAProtocol()
        if "COPSE" in config.protocols:
            self.protocols["COPSE"] = COPSEProtocol()
    
    def run_all_experiments(self) -> List[ExperimentResult]:
        """
        Run experiments on all configured models
        
        Returns:
            List of experiment results
        """
        # Load model files
        if self.config.model_files:
            model_files = self.config.model_files
        else:
            model_files = load_model_files(self.config.model_folder)
        
        print(f"Running experiments on {len(model_files)} models...")
        print(f"Protocols: {', '.join(self.config.protocols)}")
        print(f"Bit width: {self.config.bit_width}")
        print("=" * 70)
        
        for model_file in model_files:
            try:
                result = self._run_single_experiment(model_file)
                self.results.append(result)
                
                # Print summary
                self._print_result_summary(result)
                
            except Exception as e:
                print(f"ERROR running experiment for {model_file}: {e}")
                import traceback
                traceback.print_exc()
        
        print("=" * 70)
        print(f"Completed {len(self.results)}/{len(model_files)} experiments")
        
        # Save results if configured
        if self.config.save_results:
            self._save_results()
        
        return self.results
    
    def _run_single_experiment(self, model_file: str) -> ExperimentResult:
        """
        Run experiment on a single model
        
        Args:
            model_file: Path to model file
            
        Returns:
            Experiment result
        """
        result = ExperimentResult()
        result.model_name = Path(model_file).stem
        
        # Step 1: Load and convert models
        print(f"\nLoading model: {result.model_name}")
        
        copse_model = extract_copse_model(model_file)
        
        # Override precision from config if explicitly specified
        if hasattr(self.config, 'precision') and self.config.precision > 0:
            original_p = copse_model.p
            copse_model.p = self.config.precision
            print(f"  Overriding precision: {original_p} → {copse_model.p} (from config)")
        else:
            print(f"  Using auto-detected precision: p={copse_model.p}")
        
        # Sync bit_width with model's precision to ensure consistency
        self.config.bit_width = copse_model.p
        print(f"  Bit width synced to precision: {self.config.bit_width}")
        
        result.model_params = {
            "p": copse_model.p,
            "d": copse_model.d,
            "b": copse_model.b,
            "q": copse_model.q,
            "k": copse_model.k
        }
    
        
        print(f"  Parameters: p={copse_model.p}, d={copse_model.d}, b={copse_model.b}")
        
        # Convert to other model types
        vesta_model = convert_copse_to_vesta(copse_model) if "VESTA" in self.config.protocols else None
        imm_model = convert_copse_to_imm(copse_model) if "IMM" in self.config.protocols else None
        
        
        # Step 2: Generate random input
        input_data = generate_random_vector(copse_model.b, 2**copse_model.p - 1)

        # Step 3: Run each protocol
        protocol_results = {}
        
        if "IMM" in self.protocols:
            print("  Running IMM protocol...")
            imm_timing, imm_prediction = self._run_protocol(
                self.protocols["IMM"], input_data, imm_model, "IMM", copse_model
            )
            protocol_results["IMM"] = {"timing": imm_timing, "prediction": imm_prediction}
            result.add_protocol_result("IMM", imm_timing, imm_prediction)
        
        if "VESTA" in self.protocols:
            print("  Running VESTA protocol...")
            vesta_timing, vesta_prediction = self._run_protocol(
                self.protocols["VESTA"], input_data, vesta_model, "VESTA", copse_model
            )
            protocol_results["VESTA"] = {"timing": vesta_timing, "prediction": vesta_prediction}
            result.add_protocol_result("VESTA", vesta_timing, vesta_prediction)
        
        if "COPSE" in self.protocols:
            print("  Running COPSE protocol...")
            copse_timing, copse_prediction = self._run_protocol(
                self.protocols["COPSE"], input_data, copse_model, "COPSE", copse_model
            )
            protocol_results["COPSE"] = {"timing": copse_timing, "prediction": copse_prediction}
            result.add_protocol_result("COPSE", copse_timing, copse_prediction)
        
        # Step 4: Verify correctness
        if len(protocol_results) > 1:
            result.results_match = self._verify_correctness(protocol_results)
            if not result.results_match:
                print("WARNING: Results do not match across protocols!")
        
        return result
    
    def _run_protocol(self, protocol, input_data, model, protocol_name: str, 
                     base_model: COPSEModel) -> tuple:
        """
        Run a single protocol and measure timing
        
        Args:
            protocol: Protocol instance
            input_data: Input feature vector
            model: Model instance
            protocol_name: Name of protocol
            base_model: Base COPSE model (for depth calculation)
            
        Returns:
            Tuple of (timing_dict, prediction_result)
        """
        timing = {}
        
        # Calculate FHE depth
        if self.config.depth == -1:
            depth = calculate_fhe_depth(base_model.p, base_model.d, protocol_name, self.config.config_setting)
        else:
            depth = self.config.depth
        
        # 1MM tiles its MatMul into column blocks when the edge vector would not fit one rotation row
        # (N/2 slots) of the ring this depth selects; for every model that fits, nothing changes.
        if hasattr(protocol, "row_slots"):
            from ..fhe.bgv import BGVScheme
            from ..fhe.bfv import BFVScheme
            scheme_cls = BGVScheme if self.config.fhe_scheme.upper() == "BGV" else BFVScheme
            protocol.row_slots = scheme_cls.ring_dimension(depth) // 2

        # Preprocessing
        if self.config.measure_preprocessing:
            start = time.time()
            runtime_input = protocol.preprocessing(model, input_data)
            timing["preprocessing"] = (time.time() - start) * 1000  # ms
        else:
            runtime_input = protocol.preprocessing(model, input_data)
        
        # FHE scheme generation
        fhe_scheme = protocol.generate_fhe_scheme(model, depth, self.config.fhe_scheme)
        
        # Encryption
        if self.config.measure_encryption:
            start = time.time()
            ct_input = protocol.encryption(fhe_scheme, runtime_input)
            timing["encryption"] = (time.time() - start) * 1000  # ms
        else:
            ct_input = protocol.encryption(fhe_scheme, runtime_input)
        
        # Evaluation
        if self.config.measure_evaluation:
            start = time.time()
            ct_result, eval_timing = protocol.evaluate(fhe_scheme, ct_input, self.config.bit_width, return_timing=True)
            timing["evaluation"] = (time.time() - start) * 1000  # ms
            # Merge detailed evaluation timing into timing dict
            if eval_timing:
                timing.update(eval_timing)
        else:
            ct_result = protocol.evaluate(fhe_scheme, ct_input, self.config.bit_width)
        
        # Decryption
        if self.config.measure_decryption:
            start = time.time()
            result = protocol.decrypt(fhe_scheme, ct_result, model)
            timing["decryption"] = (time.time() - start) * 1000  # ms
        else:
            result = protocol.decrypt(fhe_scheme, ct_result, model)
        
        # Calculate total as sum of measured components
        encryption_time = timing.get("encryption", 0.0)
        evaluation_time = timing.get("evaluation", 0.0)
        decryption_time = timing.get("decryption", 0.0)
        timing["total"] = encryption_time + evaluation_time + decryption_time
        
        # Return both timing and prediction result
        return timing, result
    
    def _verify_correctness(self, protocol_results: Dict[str, Dict]) -> bool:
        """
        Verify that all protocols produce the same prediction results
        
        Args:
            protocol_results: Dictionary with structure:
                {
                    "IMM": {"timing": {...}, "prediction": result},
                    "VESTA": {"timing": {...}, "prediction": result},
                    "COPSE": {"timing": {...}, "prediction": result}
                }
                
        Returns:
            True if all predictions match
        """
        if len(protocol_results) < 2:
            return True  # Can't verify with only one protocol
        
        # Extract predictions
        predictions = {}
        for protocol_name, data in protocol_results.items():
            predictions[protocol_name] = data["prediction"]
        
        # Get the first prediction as reference
        protocol_names = list(predictions.keys())
        reference_prediction = predictions[protocol_names[0]]
        
        # Compare all predictions with the reference
        for protocol_name in protocol_names[1:]:
            current_prediction = predictions[protocol_name]
            
            # Compare based on prediction type
            if not self._compare_predictions(reference_prediction, current_prediction):
                print(f"  ✗ Mismatch detected: {protocol_names[0]} vs {protocol_name}")
                return False
        
        print(f"  ✓ All protocols produced matching results")
        return True
    
    def _compare_predictions(self, pred1, pred2) -> bool:
        """
        Compare two predictions for equality
        
        Args:
            pred1: First prediction
            pred2: Second prediction
            
        Returns:
            True if predictions are equal
        """
        # Handle different types of predictions
        if isinstance(pred1, list) and isinstance(pred2, list):
            # List comparison (e.g., multiple predictions)
            if len(pred1) != len(pred2):
                return False
            return all(self._compare_predictions(p1, p2) for p1, p2 in zip(pred1, pred2))
        
        elif isinstance(pred1, (int, float)) and isinstance(pred2, (int, float)):
            # Numeric comparison with tolerance for floating point
            return abs(pred1 - pred2) < 1e-6
        
        else:
            # Direct equality check
            return pred1 == pred2
    
    def _print_result_summary(self, result: ExperimentResult):
        """Print detailed timing breakdown for a single experiment result"""
        print("\n" + "=" * 90)
        print(f"Results for {result.model_name}:")
        print("=" * 90)
        
        # Part 1: Macro-level timing
        print("\nMacro-level Timing (ms):")
        print("-" * 90)
        header_macro = f"{'Protocol':<10}{'Encryption':<15}{'Evaluation':<15}{'Decryption':<15}{'Total':<12}"
        print(header_macro)
        print("-" * 90)
        
        for protocol_name in ["IMM", "VESTA", "COPSE"]:
            if protocol_name not in result.results:
                continue
            
            timing = result.results[protocol_name]
            encryption_time = timing.get("encryption", 0.0)
            evaluation_time = timing.get("evaluation", 0.0)
            decryption_time = timing.get("decryption", 0.0)
            total_time = timing.get("total", 0.0)
            
            row = f"{protocol_name:<10}{encryption_time:<15.2f}{evaluation_time:<15.2f}{decryption_time:<15.2f}{total_time:<12.2f}"
            print(row)
        
        # Part 2: Micro-level timing
        print("\nMicro-level Timing - Evaluation Phase (ms):")
        print("-" * 90)
        all_steps = ["SecComp", "Reordering", "LvlProcessing", "Reduction"]
        
        header_micro = f"{'Protocol':<10}"
        for step in all_steps:
            header_micro += f"{step:<15}"
        header_micro += f"{'Eval Total':<12}"
        print(header_micro)
        print("-" * 90)
        
        for protocol_name in ["IMM", "VESTA", "COPSE"]:
            if protocol_name not in result.results:
                continue
            
            timing = result.results[protocol_name]
            row = f"{protocol_name:<10}"
            
            # For IMM, rename MatMul to LvlProcessing for consistency
            if protocol_name == "IMM":
                sec_comp_time = timing.get("SecComp", 0.0)
                lvl_processing_time = timing.get("MatMul", 0.0)
                reordering_time = 0.0
                reduction_time = 0.0
                
                row += f"{sec_comp_time:<15.2f}"
                row += f"{reordering_time:<15.2f}"
                row += f"{lvl_processing_time:<15.2f}"
                row += f"{reduction_time:<15.2f}"
            else:
                # VESTA and COPSE use standard step names
                for step in all_steps:
                    time_ms = timing.get(step, 0.0)
                    row += f"{time_ms:<15.2f}"
            
            # Evaluation total
            eval_total = timing.get("evaluation", 0.0)
            row += f"{eval_total:<12.2f}"
            
            print(row)
        
        print("=" * 90)
        
        # Correctness check
        correctness = "✓" if result.results_match else "✗"
        print(f"Correctness: {correctness}")
    
    def _save_results(self):
        """Save results to JSON file"""
        output_path = Path(self.config.output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_results_{timestamp}.json"
        filepath = output_path / filename
        
        results_dict = [result.to_dict() for result in self.results]
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculate summary statistics across all experiments
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {}
        
        summary = {
            "num_models": len(self.results),
            "protocols": list(self.results[0].results.keys()),
            "average_times": {}
        }
        
        # Calculate average times per protocol
        for protocol in summary["protocols"]:
            times = [r.results[protocol].get("total", 0) for r in self.results]
            summary["average_times"][protocol] = {
                "mean": sum(times) / len(times),
                "min": min(times),
                "max": max(times)
            }
        
        return summary
