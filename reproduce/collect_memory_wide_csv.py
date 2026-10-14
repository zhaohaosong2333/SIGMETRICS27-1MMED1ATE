#!/usr/bin/env python3
"""Collect memory-experiment JSON logs into the WIDE per-setting CSV that
the Figure 11 memory panel + Table 9 (make_end2end.py) consume
(the wide per-model peak-RSS schema):

    Model,COPSE-<SET>,VESTA-<SET>,1MM-<SET>      (peak RSS in KB)

`scripts/run_complete_experiment.py --output-folder <dir>` writes
experiment_results_<tag>.json = [{"model_name","results":{"IMM":{...,"peak_memory_mb"},...}}].
collect_memory_csv.py already emits the LONG "Experiment,Peak Memory (MB)" form; this
companion emits the WIDE form so a fresh memory run can regenerate the main table.

Usage:
    python reproduce/collect_memory_wide_csv.py <results_dir> --setting OD --out out.csv
"""
import json
import argparse
from pathlib import Path

PROTO = {"COPSE": "COPSE", "VESTA": "VESTA", "IMM": "1MM"}  # JSON key -> CSV column prefix


def collect(json_paths):
    order, rows = [], {}
    for jp in json_paths:
        for m in json.load(open(jp)):
            name = m["model_name"]
            if not name.endswith(".mdl"):
                name = name + ".mdl"          # match the .mdl-suffixed model-name style (income_5.mdl)
            if name not in rows:
                rows[name] = {}
                order.append(name)
            for proto in ("COPSE", "VESTA", "IMM"):
                r = m.get("results", {}).get(proto)
                if not r:
                    continue
                mem_mb = r.get("peak_memory_mb")
                # MB -> KB to match the committed file's units (ratios are unit-independent).
                rows[name][PROTO[proto]] = "" if mem_mb is None else f"{mem_mb * 1024:.0f}"
    return order, rows


def main():
    ap = argparse.ArgumentParser(description="memory JSON log -> WIDE per-setting peak-memory CSV")
    ap.add_argument("results", help="a directory of experiment_results_*.json files (or one file)")
    ap.add_argument("--setting", required=True, help="OD or UD (becomes the column suffix)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p = Path(args.results)
    paths = sorted(p.glob("experiment_results_*.json")) if p.is_dir() else [p]
    if not paths:
        raise SystemExit(f"ERROR: no experiment_results_*.json under {args.results}")

    order, rows = collect(paths)
    s = args.setting
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"Model,COPSE-{s},VESTA-{s},1MM-{s}\n")
        for name in order:
            r = rows[name]
            f.write(f"{name},{r.get('COPSE','')},{r.get('VESTA','')},{r.get('1MM','')}\n")
    print(f"wrote {len(order)} rows -> {args.out}")


if __name__ == "__main__":
    main()
