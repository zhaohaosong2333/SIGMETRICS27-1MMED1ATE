#!/usr/bin/env python3
"""Generate the paper tables that are derivable from the models/compiler but weren't wired into
the figure pipeline:

  tab_modelinfo.md   paper Table 5 — evaluated-model spec (p, d, b, q) per model
  tab_modelsizes.md  paper Table (compiled vectorized model sizes) — element counts per system:
                       1MM   em          = L x 2q
                       VESTA d*ulvls      = d x (L x q)
                       COPSE reshuf+d*lvls= (q x q) + d x (L x b)

Run from repo root in the 1mm-core env (imports immed1ate).  python reproduce/make_extra_tables.py
"""
import sys, glob, argparse
from pathlib import Path
sys.path.insert(0, "src")
from immed1ate.compiler.extractor import get_unsecure_copse_model_from_serial_model as extract
from immed1ate.compiler.transformer import convert_copse_to_imm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folders", nargs="+", default=["data/models/rf", "data/models/gbdt"])
    ap.add_argument("--outdir", default="reproduce/figures_paper")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    info = ["# Table 5 — evaluated model specification (regenerated)\n",
            "| model | folder | p (precision) | d (depth) | b (branches) | q (thresholds) | L (leaves) |",
            "|---|---|--:|--:|--:|--:|--:|"]
    sizes = ["# Compiled vectorized model sizes — element counts per system (regenerated)\n",
             "| model | 1MM (em, L·2q) | VESTA (d·L·q) | COPSE (q² + d·L·b) |",
             "|---|--:|--:|--:|"]
    for folder in args.folders:
        for mdl in sorted(glob.glob(f"{folder}/*.mdl")):
            name = Path(mdl).stem
            cm = extract(mdl)
            im = convert_copse_to_imm(cm)
            L = len(im.em); p, d, b, q = cm.p, cm.d, cm.b, cm.q
            info.append(f"| {name} | {Path(folder).name} | {p} | {d} | {b} | {q} | {L} |")
            imm_sz = L * 2 * q
            vesta_sz = d * L * q
            copse_sz = q * q + d * L * b
            sizes.append(f"| {name} | {imm_sz:,} | {vesta_sz:,} | {copse_sz:,} |")

    (out / "tab_modelinfo.md").write_text("\n".join(info) + "\n")
    (out / "tab_modelsizes.md").write_text("\n".join(sizes) + "\n")
    print(f"saved {out}/tab_modelinfo.md and tab_modelsizes.md")


if __name__ == "__main__":
    main()
