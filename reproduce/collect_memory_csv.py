#!/usr/bin/env python3
"""Collect 1MMED1ATE memory-experiment JSON logs into a peak-memory CSV.

`scripts/run_complete_experiment.py --output-folder <dir>` writes JSON files
(experiment_results_<tag>.json) in the standard format (same as the speed runner):
    [{"model_name": ..., "results": {"IMM": {..., "peak_memory_mb": ...}, ...}}]

This reads those files and emits:

    Experiment,Peak Memory (MB)

with one row per {1MM|VESTA|COPSE}-<model> -- the data needed for memory plots.

Usage:
    python reproduce/collect_memory_csv.py <results_dir> --out data/output/memory.csv
"""
import sys
import json
import argparse
from pathlib import Path

PROTO_PREFIX = {"IMM": "1MM", "VESTA": "VESTA", "COPSE": "COPSE"}


def normalize_model(name):
    return name.replace("_", "")


def collect(json_paths):
    ordered, rows = [], {}
    for jp in json_paths:
        for m in json.load(open(jp)):
            model = normalize_model(m["model_name"])
            for proto in ("IMM", "VESTA", "COPSE"):
                if proto not in m.get("results", {}):
                    continue
                exp = f"{PROTO_PREFIX[proto]}-{model}"
                if exp not in rows:
                    ordered.append(exp)
                mem = m["results"][proto].get("peak_memory_mb")
                rows[exp] = mem if mem is not None else 0.0
    return ordered, rows


def main():
    ap = argparse.ArgumentParser(description="memory JSON log -> peak-memory CSV")
    ap.add_argument("results", help="a directory of experiment_results_*.json files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p = Path(args.results)
    paths = sorted(p.glob("experiment_results_*.json"))
    if not paths:
        print(f"ERROR: no experiment_results_*.json under {args.results}", file=sys.stderr)
        sys.exit(1)

    ordered, rows = collect(paths)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("Experiment,Peak Memory (MB)\n")
        for exp in ordered:
            f.write(f"{exp},{rows[exp]:.2f}\n")
    print(f"collected {len(ordered)} rows from {len(paths)} log(s) -> {args.out}")


if __name__ == "__main__":
    main()
