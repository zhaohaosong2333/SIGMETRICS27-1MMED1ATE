#!/usr/bin/env python3
"""Loaders shared by the figure scripts, and Figure 2.

  multiplicativedepth.pdf   time and peak memory of one 1024x1024 MVMul versus the BGV multiplicative
                            depth (data: <results>/motivation_matmul_depth.txt, written by the `motiv` tag)

Figure 11 and Table 9 come from make_end2end.py (which imports the loaders below),
Figure 15 and Figure 13 from make_new_figures.py, Figure 12 and Figure 14
from make_sweep_figures.py. All figures share figstyle.py.

    python reproduce/make_figures.py [--results data/paper_results] [--outdir reproduce/figures_paper]
"""
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for figstyle
import figstyle as fs                                       # shared paper style

PAPER_STAGES = fs.STAGE3                                    # bottom -> top
STAGE_COLOR = fs.STAGE3_COLOR
PROTO_ORDER = fs.PROTO_ORDER                                # left -> right
PROTO_HATCH = fs.PROTO_HATCH
CSV_PROTO = {"1MM": "IMM", "VESTA": "VESTA", "COPSE": "COPSE"}    # CSV prefix -> protocol


def load_speed_csv(path):
    """Parse a *Result.csv into {model: {protocol: {raw stage: ms}}}."""
    data = {}
    with open(path) as f:
        cols = f.readline().strip().split(",")[1:]  # Encryption,Comparison,Reordering,Level Processing,Accumulation
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            proto = CSV_PROTO.get(parts[0].split("-", 1)[0])
            if proto is None:
                continue
            model = parts[0].split("-", 1)[1]
            data.setdefault(model, {})[proto] = dict(zip(cols, [float(x) for x in parts[1:]]))
    return data


def _stage3(raw):
    """Merge the 5 raw stages into the paper's 3 (seconds)."""
    return {
        "Compare": raw.get("Comparison", 0.0) / 1000.0,
        "Path Construct": (raw.get("Reordering", 0.0) + raw.get("Level Processing", 0.0)
                           + raw.get("Accumulation", 0.0)) / 1000.0,
        "Encrypt": raw.get("Encryption", 0.0) / 1000.0,
    }


# Four bars per model: COPSE, VESTA, 1MM(UD), 1MM(OD). Baselines are ~setting-invariant
# (COPSE-OD == COPSE-UD; VESTA-OD is one limb less) so they appear once; only 1MM is shown
# at UD (matched-depth reference, cross-hatch) and at its minimal OD depth (solid, the hero).
COMBINED_ORDER = ["COPSE", "VESTA", "1MM (UD)", "1MM (OD)"]
COMBINED_HATCH = {"COPSE": PROTO_HATCH["COPSE"], "VESTA": PROTO_HATCH["VESTA"],
                  "1MM (UD)": "xxx", "1MM (OD)": ""}


def load_wide(path):
    """peak-RSS wide CSV: 'Model,COPSE-<SET>,VESTA-<SET>,1MM-<SET>' (KB) -> {model:{proto:kb}}."""
    import csv, os
    d = {}
    if not path or not os.path.exists(path):
        return d
    rows = list(csv.reader(open(path))); hdr = rows[0]
    for r in rows[1:]:
        if not r or not r[0]:
            continue
        d[r[0].replace(".mdl", "")] = {hdr[i].split("-")[0]: float(r[i])
                                       for i in range(1, len(hdr)) if r[i]}
    return d


def _gmean(vals):
    import math
    vals = [max(v, 1e-9) for v in vals]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0


def make_motivation_figure(out_path, combined_txt):
    depths, enc, comp, mem_gb = [], [], [], []
    if combined_txt and Path(combined_txt).exists():
        for line in Path(combined_txt).read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 4 and parts[0].strip().isdigit():
                depths.append(int(parts[0]))
                enc.append(float(parts[1]))
                comp.append(float(parts[2]))
                mem_gb.append(float(parts[3]) / 1024.0)  # peak MB -> GB
    if not depths:
        print(f"  [skip] multiplicativedepth.pdf: no motivation data in {combined_txt}")
        return

    # paper plots a single "Time" line (encryption + computation) + a "Memory" line
    time_s = [(enc[i] + (comp[i] if i < len(comp) else 0.0)) / 1000.0 for i in range(len(enc))]

    c_time, c_mem = fs.TOL["indigo"], fs.TOL["rose"]
    # narrow (half-column) so it can sit in a wrapfigure with native-size text -- drawing at the
    # final ~2.6in width keeps the 9pt labels readable instead of shrinking under a 0.4x rescale.
    # kept short (landscape aspect) so the whole wrapfigure clears the page-bottom footnote.
    fig, ax1 = fs.new_fig(w=fs.HALF_W, h=1.35)
    ax1.spines["top"].set_visible(True)               # close the box (figstyle hides top by default)
    ax1.plot(depths, time_s, "-s", color=c_time, markersize=3.5, label="Time")
    ax1.set_xlabel("Multiplicative depth")
    ax1.set_ylabel("Time (s)")
    handles = ax1.get_legend_handles_labels()
    if mem_gb:
        ax2 = ax1.twinx()
        ax2.plot(range(1, len(mem_gb) + 1), mem_gb, "-o", color=c_mem, markersize=3.5, label="Memory")
        ax2.set_ylabel("Memory (GB)")
        ax2.spines["top"].set_visible(True)           # top + right spines on the twin -> full frame
        ax2.spines["right"].set_visible(True)
        ax2.tick_params(length=3, width=0.6)
        h2 = ax2.get_legend_handles_labels()
        handles = (handles[0] + h2[0], handles[1] + h2[1])
    ax1.legend(handles[0], handles[1], loc="upper left")
    fs.save(fig, out_path)
    print(f"  saved {out_path.name}")



def main():
    ap = argparse.ArgumentParser(description="Figure 2 from a results directory")
    ap.add_argument("--results", default=str(REPO / "data" / "paper_results"), help="results directory")
    ap.add_argument("--outdir", default=str(REPO / "reproduce" / "figures_paper"))
    args = ap.parse_args()
    fs.use_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("[Figure 2] -> multiplicativedepth.pdf")
    make_motivation_figure(outdir / "multiplicativedepth.pdf", Path(args.results) / "motivation_matmul_depth.txt")


if __name__ == "__main__":
    main()
