#!/usr/bin/env python3
"""
Complete Experiment Script with Time and Memory Measurement

Runs all three protocols (IMM, VESTA, COPSE) on all models and outputs
both timing and memory consumption results.

Features:
- Total execution time per protocol
- Detailed step-by-step timing (SecComp, MatMul, etc.)
- Memory usage tracking
- Results saved to JSON

Usage:
    python run_complete_experiment.py --model-folder <path> [options]
"""

import sys
import argparse
import json
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))



def run_protocol_with_memory_measurement(protocol_name: str, model_file: str, 
                                        bit_width: int,
                                        config_setting: str = "OD",
                                        fhe_scheme: str = "BGV",
                                        depth: int = -1) -> Dict:
    """
    Run a single protocol in a separate process and measure memory using /usr/bin/time
    
    Args:
        protocol_name: Protocol name (IMM, VESTA, or COPSE)
        model_file: Path to model file
        bit_width: Bit width for secure comparison
        
    Returns:
        Dictionary with time_ms, peak_memory_mb, and timing details
    """
    # Get the src directory path
    src_dir = str(Path(__file__).parent.parent / "src")
    
    # Create a temporary Python script that runs only this protocol
    script_content = f'''import sys
import json
sys.path.insert(0, "{src_dir}")

from immed1ate.experiments import ExperimentConfig, ExperimentRunner

config = ExperimentConfig(
    model_files=["{model_file}"],
    protocols=["{protocol_name}"],
    bit_width={bit_width},
    config_setting="{config_setting}",
    fhe_scheme="{fhe_scheme}",
    depth={depth},
    measure_encryption=True,
    measure_evaluation=True,
    measure_decryption=True,
    save_results=False
)

runner = ExperimentRunner(config)
results = runner.run_all_experiments()

# Output results as JSON
if results:
    result = results[0]
    output = {{
        "time_ms": result.results.get("{protocol_name}", {{}}).get("total", 0),
        "details": result.results.get("{protocol_name}", {{}})
    }}
    print(json.dumps(output))
else:
    print(json.dumps({{"error": "No results"}}))
'''
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        temp_script = f.name
    
    try:
        # Run under GNU /usr/bin/time -v (Linux) to capture peak RSS.
        import shutil
        # GNU-time that supports -v (set as $GNU_TIME by reproduce/slurm/env.sh; else /usr/bin/time).
        # Without a working one we still collect timing, just no peak memory -- rather than crash.
        gnu_time = os.environ.get("GNU_TIME") or "/usr/bin/time"
        _time_flag = '-v'
        _have_time = bool(gnu_time) and (
            (os.path.isabs(gnu_time) and os.path.exists(gnu_time)) or shutil.which(gnu_time) is not None)
        time_cmd = [gnu_time, _time_flag] if _have_time else []
        
        if not _have_time:
            print("  WARNING: no GNU 'time -v' ($GNU_TIME / /usr/bin/time); peak memory -> N/A",
                  file=sys.stderr)
        result = subprocess.run(
            time_cmd + [sys.executable, temp_script],
            capture_output=True,
            text=True
        )
        
        # Parse memory from stderr
        memory_mb = None
        for line in result.stderr.split('\n'):
            # Linux GNU time: "Maximum resident set size (kbytes): 8544" (KB)
            if 'Maximum resident set size' in line:
                parts = line.strip().split()
                if parts:
                    memory_kb = int(parts[-1])
                    memory_mb = memory_kb / 1024
                    break
        
        # Parse JSON output from stdout
        timing_data = {}
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line.startswith('{'):
                try:
                    timing_data = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        
        return {
            'time_ms': timing_data.get('time_ms', 0),
            'peak_memory_mb': memory_mb,
            'details': timing_data.get('details', {})
        }
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_script):
            os.unlink(temp_script)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run complete 1MMEDIATE experiments with time and memory measurement"
    )
    
    parser.add_argument(
        "--model-folder",
        type=str,
        required=True,
        help="Path to folder containing .mdl model files"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Only these model stems of --model-folder (default: every .mdl in it)"
    )

    parser.add_argument(
        "--protocols",
        type=str,
        nargs="+",
        default=["COPSE", "VESTA", "IMM"],
        choices=["IMM", "VESTA", "COPSE"],
        help="Protocols to test (default: COPSE VESTA IMM)"
    )
    
    parser.add_argument(
        "--bit-width",
        type=int,
        default=8,
        choices=[8, 16],
        help="Bit width for secure comparison (default: 8)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for results (default: experiment_results.json)"
    )
    
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Folder to save results in standard JSON format (compatible with collect scripts)"
    )
    
    parser.add_argument(
        "--measure-memory",
        action="store_true",
        default=True,
        help="Enable memory measurement (default: enabled)"
    )
    
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable memory measurement"
    )
    
    parser.add_argument(
        "--fhe-scheme",
        type=str,
        default="BGV",
        choices=["BGV", "BFV"],
        help="FHE scheme to use (default: BGV)"
    )
    
    parser.add_argument(
        "--depth",
        type=int,
        default=-1,
        help="FHE multiplicative depth (default: auto-calculate)"
    )
    
    parser.add_argument(
        "--config-setting",
        type=str,
        default="OD",
        choices=["OD", "UD"],
        help="Configuration setting: OD (OptDepth) or UD (UniDepth) (default: OD)"
    )
    
    return parser.parse_args()


def run_complete_experiment(model_folder: str, protocols: List[str], 
                           bit_width: int, measure_memory: bool,
                           output_file: str, config_setting: str = "OD",
                           fhe_scheme: str = "BGV", depth: int = -1,
                           output_folder: str = None, models=None):
    """
    Run complete experiment with time and memory measurement
    
    Args:
        model_folder: Path to model folder
        protocols: List of protocols to test
        bit_width: Bit width for secure comparison
        measure_memory: Whether to measure memory
        output_file: Output file path
    """
    print("=" * 80)
    print("1MMEDIATE Complete Experiment - Time & Memory Measurement")
    print("=" * 80)
    print(f"Model folder: {model_folder}")
    print(f"Protocols: {', '.join(protocols)}")
    print(f"Bit width: {bit_width}")
    print(f"Memory measurement: Using /usr/bin/time (subprocess method)")
    print(f"Config setting: {config_setting}")
    print(f"FHE scheme: {fhe_scheme}")
    print(f"Depth: {'auto' if depth == -1 else depth}")
    print("=" * 80)
    
    # Load model files
    from immed1ate.experiments.config import load_model_files
    model_files = load_model_files(model_folder)
    if models:
        model_files = [f for f in model_files if Path(f).stem in set(models)]
        missing = set(models) - {Path(f).stem for f in model_files}
        if missing:
            raise SystemExit(f"models not found in {model_folder}: {sorted(missing)}")
    
    print(f"\nFound {len(model_files)} models to test\n")
    
    # Results storage
    all_results = []
    
    # Run experiments for each model
    for idx, model_file in enumerate(model_files, 1):
        print(f"\n{'='*80}")
        print(f"Model {idx}/{len(model_files)}: {Path(model_file).stem}")
        print(f"{'='*80}")
        
        model_result = {
            "model_name": Path(model_file).stem,
            "model_file": model_file,
            "protocols": {}
        }
        
        # Run each protocol separately with memory measurement
        protocol_results = {}  # Store results for table display
        
        for protocol_name in protocols:
            print(f"\n  Running {protocol_name} protocol...")
            
            
            try:
                # Run protocol in separate process with memory measurement
                proto_result = run_protocol_with_memory_measurement(
                    protocol_name, model_file, bit_width, config_setting, fhe_scheme, depth
                )
                
                # Extract data from result
                time_ms = proto_result.get('time_ms', 0)
                peak_memory_mb = proto_result.get('peak_memory_mb')
                details = proto_result.get('details', {})
                
                # Store results for table display
                protocol_results[protocol_name] = {
                    'time_ms': time_ms,
                    'peak_memory_mb': peak_memory_mb
                }
                
                # Store full results
                model_result["protocols"][protocol_name] = {
                    "time_ms": time_ms,
                    "peak_memory_mb": peak_memory_mb,
                    "memory_delta_mb": None,  # Not applicable for subprocess method
                    "details": details
                }
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                
                protocol_results[protocol_name] = {
                    'time_ms': None,
                    'peak_memory_mb': None
                }
                
                model_result["protocols"][protocol_name] = {
                    "error": str(e)
                }
        
        # Print summary table for this model
        print(f"\n  Results for {Path(model_file).stem}:")
        print("  " + "-" * 76)
        header = f"  {'Protocol':<12}{'Time (ms)':<25}{'Memory (MB)':<25}"
        print(header)
        print("  " + "-" * 76)
        
        for proto_name in protocols:
            result = protocol_results.get(proto_name, {})
            time_val = result.get('time_ms')
            mem_val = result.get('peak_memory_mb')
            
            if time_val is not None:
                time_str = f"{time_val:.2f}"
            else:
                time_str = "N/A"
            
            if mem_val is not None:
                mem_str = f"{mem_val:.2f}"
            else:
                mem_str = "N/A"
            
            print(f"  {proto_name:<12}{time_str:<25}{mem_str:<25}")
        
        print("  " + "-" * 76)
        
        all_results.append(model_result)
    
    # Calculate summary statistics
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}")
    
    summary = calculate_summary(all_results, protocols, measure_memory)
    
    # Save results
    output_data = {
        "config": {
            "model_folder": model_folder,
            "protocols": protocols,
            "bit_width": bit_width,
            "measure_memory": measure_memory,
            "fhe_scheme": fhe_scheme,
            "depth": depth
        },
        "results": all_results,
        "summary": summary
    }
    
    if output_folder is not None:
        # Save in standard format (compatible with collect scripts)
        import json as _json
        out_dir = Path(output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_tag = Path(model_folder).stem
        tag = f"{fhe_scheme}_{config_setting}_{model_tag}"
        std_path = out_dir / f"experiment_results_{tag}.json"
        std_data = []
        for mr in all_results:
            entry = {"model_name": mr["model_name"], "results": {}}
            for proto_name, proto_data in mr.get("protocols", {}).items():
                entry["results"][proto_name] = dict(proto_data.get("details", {}))
                entry["results"][proto_name]["peak_memory_mb"] = proto_data.get("peak_memory_mb")
            std_data.append(entry)
        with open(std_path, 'w') as _f:
            _json.dump(std_data, _f, indent=2)
        print(f"Standard format results saved to: {std_path}")
    
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*80}")
    
    return all_results


def calculate_summary(results: List[Dict], protocols: List[str], 
                     measure_memory: bool) -> Dict:
    """
    Calculate summary statistics
    
    Args:
        results: List of model results
        protocols: List of protocol names
        measure_memory: Whether memory was measured
        
    Returns:
        Summary statistics dictionary
    """
    summary = {
        "num_models": len(results),
        "protocols": {}
    }
    
    for protocol in protocols:
        times = []
        peak_memories = []
        
        for result in results:
            proto_result = result["protocols"].get(protocol, {})
            if "error" not in proto_result:
                times.append(proto_result.get("time_ms", 0))
                if measure_memory:
                    peak_mem = proto_result.get("peak_memory_mb")
                    if peak_mem is not None:
                        peak_memories.append(peak_mem)
        
        if times:
            stats = {
                "time": {
                    "mean_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times)
                }
            }
            
            if measure_memory and peak_memories:
                stats["memory"] = {
                    "peak_mean_mb": sum(peak_memories) / len(peak_memories),
                    "peak_min_mb": min(peak_memories),
                    "peak_max_mb": max(peak_memories)
                }
            
            summary["protocols"][protocol] = stats
    
    return summary


def main():
    """Main entry point"""
    args = parse_arguments()
    
    measure_memory = not args.no_memory
    
    output_file = args.output_file if args.output_file else "experiment_results.json"
    
    run_complete_experiment(
        model_folder=args.model_folder,
        protocols=args.protocols,
        bit_width=args.bit_width,
        measure_memory=measure_memory,
        output_file=output_file,
        config_setting=args.config_setting,
        fhe_scheme=args.fhe_scheme,
        depth=args.depth,
        output_folder=args.output_folder,
        models=args.models
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
