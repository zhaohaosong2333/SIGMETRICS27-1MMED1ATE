#!/usr/bin/env python3
"""Bit-exactness test for the numpy-vectorized compiler (mm_mul via float64 BLAS,
_generate_extended_lvls / _generate_em, diagonalize_matrix).

The reference implementations below are the original pure-Python / int64 versions, copied
verbatim. For every model the IMM extended matrix, the VESTA unified level matrices and the
diagonalized operands must be identical. No OpenFHE needed.

    python tests/test_compiler_vectorized.py                 # Category A + sweep models, b <= 1200
    python tests/test_compiler_vectorized.py --max-b 100000  # everything (slow: legacy is O(d*w*q) python)
"""
import sys
import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from immed1ate.compiler.extractor import get_unsecure_copse_model_from_serial_model as extract
from immed1ate.compiler.transformer import convert_copse_to_imm, convert_copse_to_vesta
from immed1ate.utils.matrix_ops import diagonalize_matrix


# ---- original implementations (git d36e00e / c95c471) ------------------------------------
def legacy_mm_mul(a, b):
    return (np.matmul(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)) % 2).tolist()


def legacy_mm_add(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def legacy_em(lvls, reshuf, masks):
    ulvls = [legacy_mm_mul(lvls[i].copy(), reshuf.copy()) for i in range(len(lvls))]
    elvls = []
    for level in range(len(ulvls)):
        elvl = []
        for leaf in range(len(masks[level])):
            line = []
            for element in ulvls[level][leaf]:
                if element == 0:
                    line.extend([0, 0])
                elif masks[level][leaf]:
                    line.extend([0, 1])
                else:
                    line.extend([1, 0])
            elvl.append(line)
        elvls.append(elvl)
    em = [[0 for _ in range(len(elvls[0][0]))] for _ in range(len(elvls[0]))]
    for level in range(len(elvls)):
        em = legacy_mm_add(em, elvls[level])
    return em, ulvls


def legacy_diagonalize(m):
    rownum, colnum = len(m), len(m[0])
    return [[m[row % rownum][(r + row) % colnum] for row in range(rownum)] for r in range(colnum)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-b", type=int, default=1200)
    args = ap.parse_args()
    models = sorted(REPO.glob("data/models/rf/*.mdl")) + sorted(REPO.glob("data/models/gbdt/*.mdl")) \
        + sorted(REPO.glob("data/models/sweep/*/*.mdl"))
    ok = checked = 0
    for mdl in models:
        cm = extract(str(mdl))
        if cm.b > args.max_b:
            continue
        checked += 1
        em_ref, ulvls_ref = legacy_em(cm.lvls, cm.reshuf, cm.masks)
        imm = convert_copse_to_imm(cm)
        vesta = convert_copse_to_vesta(cm)
        same = (imm.em == em_ref and vesta.ulvls == ulvls_ref
                and diagonalize_matrix(imm.em) == legacy_diagonalize(em_ref)
                and diagonalize_matrix(cm.lvls[0]) == legacy_diagonalize(cm.lvls[0])
                and diagonalize_matrix(cm.reshuf) == legacy_diagonalize(cm.reshuf))
        ok += same
        print(f"  {'OK  ' if same else 'FAIL'} {mdl.relative_to(REPO)}  (b={cm.b}, q={cm.q}, d={cm.d})")
    print(f"{ok}/{checked} models bit-identical")
    sys.exit(0 if ok == checked and checked > 0 else 1)


if __name__ == "__main__":
    main()
