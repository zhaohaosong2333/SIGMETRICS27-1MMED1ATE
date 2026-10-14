#!/usr/bin/env python3
"""Table 8: the compiled representation of representative models.

For every model with a static profile (data/sweep/static/<family>.csv, written by
scripts/gen_sweep_models.py): branches b, depth d, leaves w, the share of leaf x level entries that
are padding of shorter paths, the number of model ciphertexts each system encrypts (p + 2b for 1MM,
p + d b + d for VESTA, p + b + d b + d for COPSE) and the serialized size of that encrypted model at
each system's OD parameters. The size is the ciphertext count times the measured bytes of one
ciphertext at the (ring dimension N, multiplicative depth L) the system selects at OD, taken from the
sweep runs (<results>/sw_*.csv, column ct_bytes), where L follows calculate_fhe_depth(p, d, system).

    python reproduce/make_compiledrep_table.py [--results data/paper_results] [--outdir reproduce/figures_paper]
"""
import argparse
import csv
import glob
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from immed1ate.experiments.config import calculate_fhe_depth  # noqa: E402

PROTOS = ("IMM", "VESTA", "COPSE")
LABEL = {"IMM": "1MM", "VESTA": "VESTA", "COPSE": "COPSE"}
# the rows of the paper's table: (family, model, display name)
PAPER_ROWS = [("rf", "income", "Income (RF)"), ("gbdt", "soccer", "Soccer (GBDT)"), ("rf", "mnist", "MNIST (RF)"),
              ("full", "f8", "8 complete trees, d=8"), ("depth", "d10", "1 tree, d=10")]


def ct_bytes_table(results):
    """(N, L) -> bytes of one ciphertext (median over every sweep run at that ring and depth), and
    L -> the ring N the parameter selection picks for depth L at 128-bit security."""
    by_nl, ring_of = defaultdict(list), defaultdict(list)
    for f in glob.glob(str(Path(results) / "sw_*__*__rep*.csv")):
        if Path(f).name.startswith("sw_smoke"):
            continue
        for r in csv.DictReader(open(f)):
            try:
                N, L, cb = int(float(r["ring_dim"])), int(float(r["fhe_depth"])), float(r["ct_bytes"])
            except (KeyError, ValueError):
                continue
            by_nl[(N, L)].append(cb)
            if r.get("ring_bound", "depth") == "depth":
                ring_of[L].append(N)
    return {k: statistics.median(v) for k, v in by_nl.items()}, {L: min(v) for L, v in ring_of.items()}


def static_rows():
    rows = {}
    for f in sorted(glob.glob(str(ROOT / "data/sweep/static/*.csv"))):
        fam = Path(f).stem
        for r in csv.DictReader(open(f)):
            rows[(fam, r["model"])] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "data/paper_results"))
    ap.add_argument("--outdir", default=str(ROOT / "reproduce/figures_paper"))
    a = ap.parse_args()
    ctb, ring = ct_bytes_table(a.results)
    if not ctb:
        raise SystemExit(f"no sw_*.csv with ct_bytes under {a.results}")
    rows = static_rows()

    def line(fam, model, name):
        r = rows[(fam, model)]
        p, d, b, w = int(r["p"]), int(r["d"]), int(r["b"]), int(r["w"])
        cells = [name, f"{b:,}", str(d), f"{w:,}", f"{100 * float(r['level_pad_frac']):.1f}%"]
        cts, gbs = [], []
        for proto in PROTOS:
            n_ct = int(r[f"ct_model_{proto}"])
            L = calculate_fhe_depth(p, d, proto, "OD")
            N = ring.get(L)
            cb = ctb.get((N, L)) if N else None
            cts.append(f"{n_ct:,}")
            gbs.append(f"{n_ct * cb / 1e9:.1f}" if cb else f"n/a (no run at L={L})")
        return "| " + " | ".join(cells + cts + gbs) + " |"

    hdr = ["| Model | b | d | w | path pad. | 1MM ct | VESTA ct | COPSE ct | 1MM GB | VESTA GB | COPSE GB |",
           "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    out = ["# Table 8: compiled representation and encrypted-model size at OD\n",
           "path pad. = share of the leaf x level entries that pad shorter paths to depth d; ct = model ciphertexts "
           "(p + 2b for 1MM, p + d b + d for VESTA, p + b + d b + d for COPSE); GB = ct x measured bytes of one "
           "ciphertext at the system's OD ring and depth, in decimal GB (1e9 B).\n",
           "Bytes per ciphertext used (N, L -> MB): " +
           ", ".join(f"2^{N.bit_length() - 1}/L={L}: {v / 1e6:.2f}" for (N, L), v in sorted(ctb.items())) + "\n",
           "## The rows of the paper\n"] + hdr
    out += [line(*row) for row in PAPER_ROWS if (row[0], row[1]) in rows]
    out += ["", "## All profiled models\n"] + hdr
    out += [line(fam, m, f"{fam}/{m}") for (fam, m) in sorted(rows)]
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    (Path(a.outdir) / "tab_compiledrep.md").write_text("\n".join(out) + "\n")
    print("  saved tab_compiledrep.md")
    for row in PAPER_ROWS:
        if (row[0], row[1]) in rows:
            print("   ", line(*row))


if __name__ == "__main__":
    main()
