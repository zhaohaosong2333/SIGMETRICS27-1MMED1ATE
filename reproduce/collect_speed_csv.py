#!/usr/bin/env python3
"""Collect 1MMED1ATE speed-experiment logs into the stage-timing CSV the drawing
script consumes.

`scripts/run_speed_experiment.py` writes a per-run JSON log
(experiments/results/experiment_results_*.json). This reads that log (a file, or the
latest in a directory) and emits:

    Experiment,Encryption,Comparison,Reordering,Level Processing,Accumulation

with one row per {1MM|VESTA|COPSE}-<model> in milliseconds -- the standard stage-timing
CSV schema, so make_figures.py can plot it unchanged.

Usage:
    python reproduce/collect_speed_csv.py <log.json | results_dir> --out data/output/speed_UD.csv
"""
import sys
import json
import argparse
from pathlib import Path

PROTO_PREFIX = {"IMM": "1MM", "VESTA": "VESTA", "COPSE": "COPSE"}


def stage_values(timing):
    """Map a protocol's JSON timing dict to the 5 CSV stage columns (ms).

    IMM has no Reordering/Accumulation and reports Level Processing as 'MatMul';
    VESTA/COPSE report 'LvlProcessing'. Missing stages are 0.
    """
    return [
        timing.get("encryption", 0.0),                          # Encryption
        timing.get("SecComp", 0.0),                             # Comparison
        timing.get("Reordering", 0.0),                         # Reordering (COPSE only)
        timing.get("LvlProcessing", timing.get("MatMul", 0.0)),  # Level Processing
        timing.get("Reduction", 0.0),                          # Accumulation
    ]


def normalize_model(name):
    # The committed CSVs use income15 / soccer5 (no underscore); .mdl stems are income_15.
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
                rows[exp] = stage_values(m["results"][proto])  # later run wins
    return ordered, rows


def main():
    ap = argparse.ArgumentParser(description="speed JSON log -> stage-timing CSV")
    ap.add_argument("results", help="an experiment_results_*.json file, or a directory of them")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p = Path(args.results)
    paths = sorted(p.glob("experiment_results_*.json")) if p.is_dir() else [p]
    if not paths:
        print(f"ERROR: no experiment_results_*.json under {args.results}", file=sys.stderr)
        sys.exit(1)

    ordered, rows = collect(paths)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("Experiment,Encryption,Comparison,Reordering,Level Processing,Accumulation\n")
        for exp in ordered:
            f.write(exp + "," + ",".join(f"{v:.0f}" for v in rows[exp]) + "\n")
    print(f"collected {len(ordered)} rows from {len(paths)} log(s) -> {args.out}")


if __name__ == "__main__":
    main()
