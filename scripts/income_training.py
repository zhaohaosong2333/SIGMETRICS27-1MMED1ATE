#!/usr/bin/env python3
"""Training helpers for the Income random forests of the sensitivity study (scripts/gen_sweep_models.py):
the Census-Income data pipeline and the .mdl serializer with per-feature quantization to p bits.
"""
import numpy as np
import pandas as pd
from json import dumps
from pathlib import Path
from sklearn import preprocessing

REPO = Path(__file__).resolve().parent.parent

def load_census_income_data():
    """Load census income dataset"""
    data_path = Path(__file__).parent.parent / "data" / "train" / "census_income_weka_dataset.csv"
    print(f"Loading data from: {data_path.name}")
    df = pd.read_csv(data_path)

    for col in df:
        df[col].replace("", np.nan, inplace=True)
        df.dropna(subset=[col], inplace=True)

    print(f"Dataset shape: {df.shape}")

    label_candidates = ['income', 'class', 'target', 'label']
    label_column = None
    for candidate in label_candidates:
        if candidate in df.columns:
            label_column = candidate
            break
    if label_column is None:
        label_column = df.columns[-1]

    labels = pd.Categorical(df.pop(label_column))

    le = preprocessing.LabelEncoder()
    for col in df:
        try:
            df[col] = le.fit_transform(df[col])
        except:
            df[col] = df[col].fillna(df[col].mean() if df[col].dtype in [np.float64, np.int64] else 0)

    return df, labels


def serialize_model(model, labels, precision):
    """Serialize a Random Forest model to string format.

    Thresholds are quantized PER FEATURE: each split value is scaled by the
    maximum threshold observed *for that feature* across the whole forest,
    onto the integer grid [0, precision-1].

    Previously a single global `thresh_max` was shared across all features:
        scaled = int(threshold * (precision / thresh_max))
    Because the census columns are LabelEncoded onto very different ranges
    (e.g. `fnlwgt` -> ~26k distinct codes vs most features < ~120), whichever
    forest happened to split on a high-cardinality feature drove `thresh_max`
    to ~26k, and every threshold below thresh_max/precision then rounded to 0.
    Those zero thresholds were stripped by the `unsecure` optimizer, so the
    number of surviving thresholds q (and hence the single-matmul dimension 2q,
    which dominates IMM/VESTA encryption + evaluation time) collapsed for some
    models and not others -- producing spurious order-of-magnitude swings in
    the depth/precision sweeps. Per-feature scaling keeps q faithful to the
    real branch count, so timings reflect model size rather than a quantization
    artifact.
    """
    # Per-feature max split threshold across the forest (branch nodes only;
    # leaf nodes carry the sentinel threshold -2 and are skipped).
    feat_max = {}
    for dec_tree in model.estimators_:
        t = dec_tree.tree_
        for node_idx in range(t.node_count):
            if t.children_left[node_idx] != t.children_right[node_idx]:
                f = int(t.feature[node_idx])
                if t.threshold[node_idx] > feat_max.get(f, 0.0):
                    feat_max[f] = float(t.threshold[node_idx])

    def dfs(node_idx, tree, dec_tree):
        if tree.children_left[node_idx] != tree.children_right[node_idx]:
            f = int(tree.feature[node_idx])
            scaled_thresh = int(tree.threshold[node_idx] * ((precision - 1) / feat_max[f]))
            left_str = dfs(tree.children_left[node_idx], tree, dec_tree)
            right_str = dfs(tree.children_right[node_idx], tree, dec_tree)
            return "Branch {threshold = Threshold {feat = F %d, val = %d}, left = %s, right = %s}" % (
                tree.feature[node_idx], scaled_thresh, left_str, right_str)
        else:
            predicted_class = dec_tree.classes_[np.argmax(tree.value[node_idx])]
            return "Label (L %d)" % predicted_class

    output_lines = [dumps(list(labels.categories))]
    for dec_tree in model.estimators_:
        tree = dec_tree.tree_
        output_lines.append(dfs(0, tree, dec_tree))
    return "\n".join(output_lines)


