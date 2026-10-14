#!/usr/bin/env python3
"""
Main Experiment Script

Runs speed comparison experiments across different protocols.
Usage:
    python run_experiment.py --model-folder <path> [--protocols IMM VESTA COPSE]
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from immed1ate.experiments import ExperimentConfig, ExperimentRunner


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run 1MMEDIATE protocol comparison experiments"
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
        default=["IMM", "VESTA", "COPSE"],
        choices=["IMM", "VESTA", "COPSE"],
        help="Protocols to test (default: all)"
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
    
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs per model for averaging (default: 1)"
    )
    
    parser.add_argument(
        "--output-folder",
        type=str,
        default="experiments/results",
        help="Folder to save results (default: experiments/results)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to file"
    )
    
    return parser.parse_args()


def main():
    """Main experiment runner"""
    args = parse_arguments()
    
    # Create configuration
    model_files = []
    if args.models:
        model_files = [str(Path(args.model_folder) / f"{m}.mdl") for m in args.models]
        for f in model_files:
            if not Path(f).exists():
                raise SystemExit(f"model not found: {f}")
    config = ExperimentConfig(
        model_folder=args.model_folder,
        model_files=model_files,
        protocols=args.protocols,
        fhe_scheme=args.fhe_scheme,
        depth=args.depth,
        config_setting=args.config_setting,
        num_runs=args.num_runs,
        save_results=not args.no_save,
        output_folder=args.output_folder,
        measure_preprocessing=False,
        measure_encryption=True,
        measure_evaluation=True,
        measure_decryption=True  # Enable decryption timing
    )
    
    print("=" * 70)
    print("1MMEDIATE Protocol Comparison Experiment")
    print("=" * 70)
    print(f"Model folder: {args.model_folder}")
    print(f"Protocols: {', '.join(args.protocols)}")
    print(f"FHE scheme: {args.fhe_scheme}")
    print(f"Depth: {'auto' if args.depth == -1 else args.depth}")
    print(f"Config setting: {args.config_setting}")
    print("=" * 70)
    
    # Create and run experiment
    runner = ExperimentRunner(config)
    results = runner.run_all_experiments()
    
    # Print summary statistics across all models (if multiple models)
    if results and len(runner.results) > 1:
        summary = runner.get_summary_statistics()
        
        print("\n" + "=" * 90)
        print("Summary Statistics Across All Models")
        print("=" * 90)
        print(f"Total models tested: {summary['num_models']}")
        print(f"Protocols: {', '.join(summary['protocols'])}")
        print("\nAverage Total Time (ms):")
        
        for protocol, stats in summary['average_times'].items():
            print(f"  {protocol:<10}: mean={stats['mean']:.2f}, "
                  f"min={stats['min']:.2f}, max={stats['max']:.2f}")
        
        print("=" * 90)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
