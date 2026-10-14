#!/usr/bin/env python3
"""Flatten a sweep results file (scripts/run_sweep_experiment.py, one JSON object per run) into
one CSV row per (model, protocol, setting/depth), joined with the model's static profile
(data/sweep/static/<family>.csv).

    python reproduce/collect_sweep_csv.py <results>/sweep_raw/sw_ens_od__<label>__rep1.jsonl \
        --out <results>/sw_ens_od__<label>__rep1.csv
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COLS = [
    "family", "model", "protocol", "setting", "fhe_depth", "ring_dim", "ring_dim_security",
    "ring_bound", "log2_Q", "row_slots_needed", "p", "d", "b", "q", "w",
    # timing (ms): the paper's end-to-end = encryption + evaluation + decryption
    "total", "encryption", "evaluation", "decryption", "SecComp", "MatMul", "Reordering",
    "LvlProcessing", "Reduction",
    # model-owner compile / matrix construction (ms) and key generation
    "t_extract_ms", "t_transform_ms", "t_preprocess_ms", "t_keygen_ms",
    # memory
    "peak_rss_kb", "hwm_keygen_kb", "hwm_encrypted_kb", "hwm_evaluated_kb", "rss_preprocessed_kb",
    "rss_encrypted_kb",
    # data volume
    "ct_model", "ct_client", "ct_bytes", "bytes_model_ct", "bytes_client_ct", "bytes_key_mult",
    "bytes_key_rot",
    # homomorphic op counts / time inside the evaluation
    "eval_rot_n", "eval_rot_ms", "eval_mult_n", "eval_mult_ms", "eval_add_n", "eval_add_ms",
    "matmul_rot_n", "matmul_mult_n", "matmul_ms", "seccomp_mult_n", "encrypt_n", "encode_ms",
    "encrypt_ms",
    "imm_tiles", "imm_tile_width",
    "correct", "mismatch_leaves", "n_trees_won", "imm_counts_exact", "legacy_matmul",
    "returncode", "wall_s", "error",
]
STATIC = ["n_trees", "density", "leaf_depth_mean", "tree_depth_mean", "level_pad_frac",
          "zero_frac_IMM", "zero_frac_VESTA", "zero_frac_COPSE", "elem_IMM", "elem_VESTA",
          "elem_COPSE", "nnz_IMM", "cfg_n", "cfg_depth", "cfg_leaves", "cfg_p", "cfg_complete"]


def ops(rec, stages, op, field):
    return sum(v for k, v in rec.items()
               if k.startswith("op_") and k.endswith(f"_{op}_{field}") and k.split("_")[1] in stages)


def flatten(rec):
    row = {k: rec.get(k, "") for k in COLS}
    mem = rec.get("mem_kb", {})
    row["hwm_keygen_kb"] = mem.get("keygen", {}).get("VmHWM", "")
    row["hwm_encrypted_kb"] = mem.get("encrypted", {}).get("VmHWM", "")
    row["hwm_evaluated_kb"] = mem.get("evaluated", {}).get("VmHWM", "")
    row["rss_preprocessed_kb"] = mem.get("preprocessed", {}).get("VmRSS", "")
    row["rss_encrypted_kb"] = mem.get("encrypted", {}).get("VmRSS", "")
    if "error" not in rec:
        ev = ("SecComp", "MatMul", "LevelOp", "Eval")
        row.update(eval_rot_n=ops(rec, ev, "EvalRotate", "n"), eval_rot_ms=ops(rec, ev, "EvalRotate", "ms"),
                   eval_mult_n=ops(rec, ev, "EvalMult", "n"), eval_mult_ms=ops(rec, ev, "EvalMult", "ms"),
                   eval_add_n=ops(rec, ev, "EvalAdd", "n"), eval_add_ms=ops(rec, ev, "EvalAdd", "ms"),
                   matmul_rot_n=ops(rec, ("MatMul",), "EvalRotate", "n"),
                   matmul_mult_n=ops(rec, ("MatMul",), "EvalMult", "n"),
                   matmul_ms=sum(c["ms"] for c in rec.get("matmul_calls", [])),
                   seccomp_mult_n=ops(rec, ("SecComp",), "EvalMult", "n"),
                   encrypt_n=ops(rec, ("Encrypt",), "Encrypt", "n"),
                   encode_ms=ops(rec, ("Encrypt",), "MakePackedPlaintext", "ms"),
                   encrypt_ms=ops(rec, ("Encrypt",), "Encrypt", "ms"))
    if "error" in rec:
        row["error"] = rec["error"].strip().splitlines()[-1][:200] if rec["error"].strip() else "?"
    return row


def static_rows(family):
    path = ROOT / "data/sweep/static" / f"{family}.csv"
    if not path.exists():
        return {}
    return {r["model"]: r for r in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for path in args.jsonl:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rows.append(flatten(json.loads(line)))
    cache = {}
    for row in rows:
        fam = row["family"]
        if fam not in cache:
            cache[fam] = static_rows(fam)
        st = cache[fam].get(row["model"], {})
        for k in STATIC:
            row[k] = st.get(k, "")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS + STATIC)
        w.writeheader()
        w.writerows(rows)
    bad = sum(1 for r in rows if r["correct"] is not True)
    print(f"collected {len(rows)} runs -> {out}  ({bad} failed or incorrect)")


if __name__ == "__main__":
    main()
