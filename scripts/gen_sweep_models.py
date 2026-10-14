#!/usr/bin/env python3
"""
Generate the model families for the extended sensitivity / characterization sweep
(memory and runtime under varying tree depth, node count, ensemble size and precision).

Every family is an Income-trained random forest (same data prep, split, quantization and
RandomForestClassifier settings as scripts/income_training.py), with exactly ONE
structural knob varied per family:

  ens    ensemble size n = 4..128 trees, each tree d<=6 / <=32 leaves   (per-tree shape fixed;
         n<=64 is the training config of the existing Figure 12(a) "b" sweep)
  size   per-tree size at fixed n=8, d=8: <=16..256 leaves               (tree density 8%..58%)
  depth  depth d = 5..10 at fixed b~248 (more, shallower trees <-> fewer, deeper trees;
         the training config of the existing Figure 12(c) sweep)
  full   complete (density 1) trees, n=8, depth d = 3..8: each trained tree is padded to a
         complete binary tree with label-preserving "ghost" splits (the paper's 7.2 mitigation)
  prec   precision p in {8,16} x n in {8,32}  (d<=6, <=32 leaves)
  fhe    two ens models (n=16, n=64) reused for the FHE-parameter (multiplicative depth) sweep

Thresholds that quantize to 0 are clamped to 1 (x < 0 can never fire, and the compiler's
`unsecure` pass drops zero-threshold slots, which silently removes that branch's right edge from
the 1MM edge count). Every sweep model therefore has q == b.

Writes  data/models/sweep/<family>/<name>.mdl  and  data/sweep/static/<family>.csv
(hardware-independent structure + compiled-representation profile, see
src/immed1ate/utils/characterize.py::static_profile).

    python scripts/gen_sweep_models.py                      # all families
    python scripts/gen_sweep_models.py --families ens size  # subset
    python scripts/gen_sweep_models.py --profile-only data/models/rf data/models/gbdt
"""

import re
import sys
import csv
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from income_training import load_census_income_data, serialize_model  # noqa: E402

RF_FIXED = dict(random_state=41599, max_features="sqrt",
                min_samples_split=2, min_samples_leaf=1, n_jobs=-1)

FAMILIES = {
    "ens":   [dict(name=f"n{n:03d}", n=n, depth=6, leaves=32, p=8)
              for n in (4, 8, 16, 32, 64, 128)],
    "size":  [dict(name=f"l{L:03d}", n=8, depth=8, leaves=L, p=8)
              for L in (16, 32, 64, 128, 256)],
    "depth": [dict(name=f"d{d:02d}", n=n, depth=d, leaves=L, p=8)
              for d, n, L in ((5, 32, 8), (6, 16, 16), (7, 8, 32),
                              (8, 4, 64), (9, 2, 128), (10, 1, 256))],
    "full":  [dict(name=f"f{d}", n=8, depth=d, leaves=None, p=8, complete=True)
              for d in (3, 4, 5, 6, 7, 8)],
    "prec":  [dict(name=f"n{n:03d}p{p:02d}", n=n, depth=6, leaves=32, p=p)
              for n in (8, 32) for p in (8, 16)],
    "fhe":   [dict(name=f"n{n:03d}", n=n, depth=6, leaves=32, p=8) for n in (16, 64)],
}


def prepare_training_data():
    """70/30 stratified split (seed 42) and mean imputation."""
    from sklearn.model_selection import train_test_split
    df, labels = load_census_income_data()
    train_data, _, train_labels, _ = train_test_split(
        df, labels, stratify=labels, test_size=0.3, random_state=42)
    train_data = train_data.fillna(train_data.mean())
    return train_data, train_labels, labels


def _node_text(node):
    if node.is_leaf():
        return "Label (L %s)" % node.label_id
    return "Branch {threshold = Threshold {feat = F %s, val = %s}, left = %s, right = %s}" % (
        node.feature_id, node.thresh, _node_text(node.left), _node_text(node.right))


def complete_trees(serialized, depth, p, seed=7):
    """Pad every tree to a complete binary tree of `depth` levels. A leaf above the bottom
    level becomes a ghost Branch on a feature the forest already uses, with a random non-zero
    threshold and the leaf's label on both sides, so every prediction is unchanged."""
    import numpy as np
    from immed1ate.compiler.extractor import create_node
    from immed1ate.tree.nodes import Branch, Leaf

    rng = np.random.default_rng(seed)
    lines = serialized.split("\n")
    roots = [create_node(line) for line in lines[1:]]
    feats = sorted({br for br in re.findall(r"feat = F (\d+)", serialized)}, key=int)

    def pad(node, level):
        if level == depth:
            return node
        if node.is_leaf():
            ghost = Branch(feats[rng.integers(len(feats))], str(int(rng.integers(1, 2 ** p))), level)
            ghost.set_left(Leaf(node.label_id))
            ghost.set_right(Leaf(node.label_id))
            node = ghost
        node.set_left(pad(node.left, level + 1))
        node.set_right(pad(node.right, level + 1))
        return node

    return "\n".join([lines[0]] + [_node_text(pad(r, 0)) for r in roots])


def train_and_serialize(spec, train_data, train_labels, labels):
    """Returns (serialized model, number of zero thresholds clamped to 1)."""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=spec["n"], max_depth=spec["depth"],
                                   max_leaf_nodes=spec["leaves"], **RF_FIXED)
    model.fit(train_data, train_labels)
    text = serialize_model(model, labels, 1 << spec["p"])
    text, n_clamped = re.subn(r"val = 0\}", "val = 1}", text)
    if spec.get("complete"):
        text = complete_trees(text, spec["depth"], spec["p"])
    return text, n_clamped


def write_profiles(rows, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  profile -> {out_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--models-root", default=str(ROOT / "data/models/sweep"))
    ap.add_argument("--static-root", default=str(ROOT / "data/sweep/static"))
    ap.add_argument("--profile-only", nargs="+", metavar="FOLDER",
                    help="only write static profiles for existing .mdl folders (e.g. Category A)")
    args = ap.parse_args()

    from immed1ate.utils.characterize import static_profile

    if args.profile_only:
        for folder in args.profile_only:
            folder = Path(folder)
            rows = []
            for mdl in sorted(folder.glob("*.mdl")):
                print(f"  profiling {mdl}")
                rows.append({"family": folder.name, **static_profile(str(mdl))})
            write_profiles(rows, Path(args.static_root) / f"{folder.name}.csv")
        return

    train_data, train_labels, labels = prepare_training_data()
    for fam in args.families:
        print(f"\n== family {fam} ==")
        fam_dir = Path(args.models_root) / fam
        fam_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for spec in FAMILIES[fam]:
            path = fam_dir / f"{spec['name']}.mdl"
            text, n_clamped = train_and_serialize(spec, train_data, train_labels, labels)
            path.write_text(text)
            prof = static_profile(str(path))
            rows.append({"family": fam, "cfg_n": spec["n"], "cfg_depth": spec["depth"],
                         "cfg_leaves": spec["leaves"] or "", "cfg_p": spec["p"],
                         "cfg_complete": int(bool(spec.get("complete"))),
                         "zero_thresh_clamped": n_clamped, **prof})
            print(f"  {spec['name']}: n={prof['n_trees']} b={prof['b']} q={prof['q']} "
                  f"w={prof['w']} d={prof['d']} density={prof['density']:.3f} "
                  f"zero_thresh={prof['zero_thresh']}")
        write_profiles(rows, Path(args.static_root) / f"{fam}.csv")


if __name__ == "__main__":
    main()
