#!/usr/bin/env python3
"""Train the Category-A (real-world, naturally-trained) tree models for 1MMED1ATE.

Category A is the paper's HEADLINE suite: for each of FOUR real datasets
(Income, Soccer, Iris, MNIST) we train a Random Forest and a Gradient-Boosted
Decision Tree with a single fixed, sensible hyper-parameter configuration -- NOT
an artificial structural sweep. The models are serialized to the `.mdl` DSL the
FHE backend consumes:

    line 1        : JSON list of class labels (cosmetic -- the FHE circuit is
                    built from branch structure only; see src/immed1ate/tree/
                    random_forest.py which never reads the leaf label id).
    lines 2..N+1  : one tree per line, fully-parenthesized
                    Branch {threshold = Threshold {feat = F i, val = t}, left=.., right=..}
                    Leaf   Label (L k)

Thresholds are quantized PER FEATURE onto the integer grid [0, 2^p - 1] (the
"fixed" quantization of scripts/income_training.py). RF leaves carry the real
predicted class; GBDT leaves are the structural sentinel `L 0` -- boosting stage
trees are regressors whose per-leaf value is a residual increment, not a class
vote, so (matching the existing data/models/gbdt/*.mdl) only their structure and
thresholds feed the benchmark.

Precision: all Category-A models use p=16 (matching Income/Soccer). The FHE
extractor infers p from the data (max threshold > 255 -> 16-bit), so no filename
convention is required.

Usage:
    python scripts/train_category_a.py --rf-out data/models/rf \
        --gbdt-out data/models/gbdt --micro-out data/models/micro \
        [--datasets income soccer iris mnist] [--mnist-cache <path>]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 41599
SPLIT_STATE = 42
BIT_WIDTH = 16          # p = 16 for all Category-A models

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Serialization (shared per-feature quantization)                             #
# --------------------------------------------------------------------------- #
def _forest_trees(model):
    """Return the flat list of underlying sklearn Tree objects.

    RandomForest: one per estimator.
    GradientBoosting: estimators_ is (n_stages, n_classes) of regressors;
    flatten row-major (matches the existing data/models/gbdt/*.mdl layout).
    """
    if isinstance(model, RandomForestClassifier):
        return [(est.tree_, est) for est in model.estimators_]
    if isinstance(model, GradientBoostingClassifier):
        out = []
        for stage in model.estimators_:
            for reg in stage:
                out.append((reg.tree_, None))   # None -> GBDT leaves are L 0
        return out
    raise TypeError(type(model))


def _feat_max(trees):
    """Per-feature maximum split threshold across the whole forest."""
    fmax = {}
    for tree, _ in trees:
        for n in range(tree.node_count):
            if tree.children_left[n] != tree.children_right[n]:
                f = int(tree.feature[n])
                if tree.threshold[n] > fmax.get(f, 0.0):
                    fmax[f] = float(tree.threshold[n])
    return fmax


def serialize(model, class_labels, precision):
    """Serialize an RF or GBDT model to the .mdl DSL string."""
    trees = _forest_trees(model)
    fmax = _feat_max(trees)

    def dfs(tree, dec_est, node):
        if tree.children_left[node] != tree.children_right[node]:
            f = int(tree.feature[node])
            denom = fmax[f] if fmax.get(f, 0.0) > 0 else 1.0
            val = int(tree.threshold[node] * ((precision - 1) / denom))
            left = dfs(tree, dec_est, tree.children_left[node])
            right = dfs(tree, dec_est, tree.children_right[node])
            return ("Branch {threshold = Threshold {feat = F %d, val = %d}, "
                    "left = %s, right = %s}" % (f, val, left, right))
        # leaf
        if dec_est is None:                       # GBDT stage tree -> structural sentinel
            return "Label (L 0)"
        cls = dec_est.classes_[int(np.argmax(tree.value[node]))]
        return "Label (L %d)" % int(cls)

    lines = [json.dumps(list(class_labels))]
    for tree, dec_est in trees:
        lines.append(dfs(tree, dec_est, 0))
    return "\n".join(lines) + "\n"


def mdl_stats(model, precision):
    """Return (#trees_in_mdl, total_branches, total_leaves, max_depth, #feats_used)."""
    trees = _forest_trees(model)
    nb = nl = 0
    md = 0
    feats = set()
    for tree, _ in trees:
        md = max(md, int(tree.max_depth))
        for n in range(tree.node_count):
            if tree.children_left[n] != tree.children_right[n]:
                nb += 1
                feats.add(int(tree.feature[n]))
            else:
                nl += 1
    return len(trees), nb, nl, md, len(feats)


# --------------------------------------------------------------------------- #
# Dataset loaders  ->  (X: np.ndarray[int/float], y_codes: np.ndarray[int],   #
#                       class_labels: list, name, n_features, n_classes)       #
# --------------------------------------------------------------------------- #
def load_income():
    csv = REPO / "data/train/census_income_weka_dataset.csv"
    df = pd.read_csv(csv)
    df = df.replace("", np.nan)
    label_col = df.columns[-1]                    # income_level
    labels_raw = df.pop(label_col).astype(str).str.strip()
    # census classes: normalize to the canonical <=50K / >50K spelling
    labels_raw = labels_raw.str.replace(".", "", regex=False)
    for col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    y = LabelEncoder().fit(["<=50K", ">50K"])
    yc = y.transform(labels_raw)
    return df.values.astype(float), yc, ["<=50K", ">50K"], "income", df.shape[1], 2


def load_soccer():
    csv = REPO / "data/train/soccer_international_history_scikit_label_dataset.csv"
    df = pd.read_csv(csv)
    target = "home_team_result"
    labels_raw = df.pop(target)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df = df.fillna(0)
    classes = sorted(labels_raw.unique().tolist())
    code = {c: i for i, c in enumerate(classes)}
    yc = labels_raw.map(code).values.astype(int)
    return df.values.astype(float), yc, [float(c) for c in classes], "soccer", df.shape[1], len(classes)


def load_iris():
    from sklearn.datasets import load_iris as _li
    d = _li()
    return (d.data.astype(float), d.target.astype(int),
            list(d.target_names), "iris", d.data.shape[1], len(d.target_names))


def load_mnist(cache=None):
    from sklearn.datasets import fetch_openml
    if cache:
        os.environ["SCIKIT_LEARN_DATA"] = str(cache)
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    X = X.astype(float)
    y = y.astype(int)
    classes = list(range(10))
    return X, y, classes, "mnist", X.shape[1], 10


# --------------------------------------------------------------------------- #
# Per-dataset model configs (natural; MNIST capped for FHE tractability)      #
# --------------------------------------------------------------------------- #
CONFIGS = {
    #            RF: (n_estimators, max_depth)   GBDT: (n_estimators, max_depth, lr)
    "income": dict(rf=(15, 6),  gbdt=(30, 5, 0.1)),
    "soccer": dict(rf=(15, 6),  gbdt=(30, 5, 0.1)),
    "iris":   dict(rf=(10, 4),  gbdt=(20, 3, 0.1)),
    "mnist":  dict(rf=(15, 7),  gbdt=(15, 3, 0.1)),   # depth/tree-capped so b stays ~1500-2000
}

LOADERS = {"income": load_income, "soccer": load_soccer,
           "iris": load_iris, "mnist": load_mnist}


def train_one(name, Xtr, ytr, Xte, yte, class_labels, kind):
    cfg = CONFIGS[name]
    if kind == "rf":
        n, d = cfg["rf"]
        model = RandomForestClassifier(
            n_estimators=n, max_depth=d, random_state=RANDOM_STATE,
            max_features="sqrt", min_samples_split=2, min_samples_leaf=1, n_jobs=-1)
    else:
        n, d, lr = cfg["gbdt"]
        model = GradientBoostingClassifier(
            n_estimators=n, max_depth=d, learning_rate=lr, random_state=RANDOM_STATE)
    model.fit(Xtr, ytr)
    tr_acc = model.score(Xtr, ytr)
    te_acc = model.score(Xte, yte)
    return model, tr_acc, te_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf-out", default=str(REPO / "data/models/rf"))
    ap.add_argument("--gbdt-out", default=str(REPO / "data/models/gbdt"))
    ap.add_argument("--micro-out", default=str(REPO / "data/models/micro"))
    ap.add_argument("--datasets", nargs="+", default=["income", "soccer", "iris", "mnist"])
    ap.add_argument("--mnist-cache", default=str(REPO / "data/train/mnist_cache"))
    ap.add_argument("--no-micro", action="store_true")
    args = ap.parse_args()

    precision = 1 << BIT_WIDTH
    rf_dir, gb_dir = Path(args.rf_out), Path(args.gbdt_out)
    rf_dir.mkdir(parents=True, exist_ok=True)
    gb_dir.mkdir(parents=True, exist_ok=True)

    specs = []
    for name in args.datasets:
        print(f"\n=== {name} ===", flush=True)
        loader = LOADERS[name]
        X, y, class_labels, _, nfeat, ncls = (
            loader(args.mnist_cache) if name == "mnist" else loader())
        strat = y if np.min(np.bincount(y)) >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=SPLIT_STATE, stratify=strat)
        print(f"  loaded: {X.shape[0]} samples, {nfeat} feats, {ncls} classes", flush=True)

        for kind, out_dir in (("rf", rf_dir), ("gbdt", gb_dir)):
            model, tr_acc, te_acc = train_one(name, Xtr, ytr, Xte, yte, class_labels, kind)
            ser = serialize(model, class_labels, precision)
            (out_dir / f"{name}.mdl").write_text(ser)
            nt, nb, nl, md, nf = mdl_stats(model, precision)
            specs.append(dict(dataset=name, kind=kind.upper(), trees=nt, p=BIT_WIDTH,
                              b=nb, leaves=nl, d=md, feats=nf, nfeat_total=nfeat,
                              classes=ncls, train_acc=round(tr_acc, 4),
                              test_acc=round(te_acc, 4)))
            print(f"  {kind.upper():4} trees={nt:4} b={nb:5} leaves={nl:5} d={md:2} "
                  f"feats_used={nf:4} test_acc={te_acc:.4f} -> {out_dir}/{name}.mdl", flush=True)

    # ------- micro models (small real-Iris RFs, b<=50) for the smoke tests and the SEAL backend check -------
    # 8-bit precision keeps them cheap on every backend.
    micro_precision = 1 << 8
    if not args.no_micro and "iris" in args.datasets:
        micro_dir = Path(args.micro_out)
        micro_dir.mkdir(parents=True, exist_ok=True)
        X, y, class_labels, _, nfeat, ncls = load_iris()
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=SPLIT_STATE, stratify=y)
        print(f"\n=== micro (micro Iris RF suite, p=8, b<=50) ===", flush=True)
        for n_est, depth in [(1, 4), (2, 4), (3, 5)]:
            m = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                       random_state=RANDOM_STATE, max_features="sqrt", n_jobs=-1)
            m.fit(Xtr, ytr)
            ser = serialize(m, class_labels, micro_precision)
            # No underscore in the name (collect_speed_csv strips underscores from model keys);
            # the branch count makes the name unique.
            nb_tmp = mdl_stats(m, micro_precision)[1]
            mname = f"irisrf{nb_tmp}"
            (micro_dir / f"{mname}.mdl").write_text(ser)
            nt, nb, nl, md, nf = mdl_stats(m, micro_precision)
            specs.append(dict(dataset="iris(micro)", kind="RF", trees=nt, p=8,
                              b=nb, leaves=nl, d=md, feats=nf, nfeat_total=nfeat,
                              classes=ncls, train_acc=round(m.score(Xtr, ytr), 4),
                              test_acc=round(m.score(Xte, yte), 4)))
            print(f"  {mname:12} trees={nt} b={nb} leaves={nl} d={md} "
                  f"test_acc={m.score(Xte,yte):.4f}", flush=True)

    # ------- write spec table -------
    df = pd.DataFrame(specs)
    out_csv = REPO / "data/models/category_a_specs.csv"
    df.to_csv(out_csv, index=False)
    print("\n===== Category-A model specs =====")
    print(df.to_string(index=False))
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
