#!/usr/bin/env python3
"""Export preprocessed 1MMED1ATE tensors (per .mdl) to JSON for the SEAL C++ backend.

For each model we run the SAME extractor/transformer/preprocessing path the OpenFHE
runner uses (so the C++ backend encrypts byte-identical plaintext tensors), plus a
plaintext "golden" winning-leaf vector computed from the IMM integer circuit. All three
protocols provably select the same leaf (runner._verify_correctness), so one golden
validates every bench: each bench asserts (decrypted_leaf == answer) == golden, where
answer = d for IMM (edge-count) and 1 for COPSE/VESTA (edge-product).

Output: data/exported/<name>.json  (compact). Consumed by seal_backend/*_bench.

Usage:
  python scripts/export_tensors.py --model-folder data/models/rf [data/models/gbdt] \
         --out-dir data/exported [--validate]
"""
import sys, os, json, random, argparse, glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from immed1ate.compiler.extractor import get_unsecure_copse_model_from_serial_model as extract
from immed1ate.compiler.transformer import convert_copse_to_imm, convert_copse_to_vesta
from immed1ate.runtime.imm_protocol import IMMProtocol
from immed1ate.runtime.vesta_protocol import VESTAProtocol
from immed1ate.runtime.copse_protocol import COPSEProtocol
from immed1ate.utils.encoding import generate_random_vector


def imm_golden(im, cm, input_data):
    """Plaintext IMM circuit -> winning-leaf boolean vector (length L).
    decisions[j] = 1 if dual_x[j] < dthresh[j] else 0 ; leaf value = #satisfied edges;
    winning leaf value == d (all d edges on its path satisfied)."""
    p, d = cm.p, cm.d
    dual_x = []
    for x in input_data:
        dual_x.append(x)
        dual_x.append((2 ** p) - 1 - x)
    dth = im.dthresh
    n = len(dth)                                   # 2q
    dec = [1 if dual_x[j] < dth[j] else 0 for j in range(n)]
    leafs = [sum(im.em[i][j] * dec[j] for j in range(n)) for i in range(len(im.em))]
    return [1 if v == d else 0 for v in leafs], leafs


def export_model(mdl_path, out_dir, validate=False):
    name = Path(mdl_path).stem
    cm = extract(mdl_path)
    im = convert_copse_to_imm(cm)
    vm = convert_copse_to_vesta(cm)
    p, d, b, q, k = cm.p, cm.d, cm.b, cm.q, cm.k
    L = len(im.em)

    # Deterministic per-model query (C++ consumes the dumped bit-matrices, not the seed).
    random.seed(f"1mm-{name}")
    inp = generate_random_vector(b, 2 ** p - 1)

    imm_ri   = IMMProtocol().preprocessing(im, inp)     # d_facts[p][2q], d_thresh[p][2q], em[2q][L]
    vesta_ri = VESTAProtocol().preprocessing(vm, inp)   # facts[p][q], thresh[p][q], u_lvls[d][q][L], masks[d][L]
    copse_ri = COPSEProtocol().preprocessing(cm, inp)   # facts[p][q], thresh[p][q], reshuf[q][q], lvls[d][b][L], masks[d][L]

    golden, leaf_vals = imm_golden(im, cm, inp)

    obj = {
        "name": name, "p": p, "d": d, "b": b, "q": q, "k": k, "L": L,
        "golden": golden,                                   # length L, 0/1 winning-leaf vector
        "imm": {
            "answer": d,
            "sc_length": len(imm_ri.em),                    # 2q (ones-plaintext length for sec_comp)
            "mm_colnum": len(imm_ri.em),                    # 2q
            "dfacts_bits": imm_ri.d_facts,                  # p x 2q
            "dthresh_bits": imm_ri.d_thresh,                # p x 2q
            "em_diag": imm_ri.em,                           # 2q x L (diagonalized)
        },
        "vesta": {
            "answer": 1,
            "sc_length": len(vesta_ri.u_lvls[0]),           # q
            "mm_colnum": len(vesta_ri.u_lvls[0]),           # q
            "facts_bits": vesta_ri.facts,                   # p x q
            "thresh_bits": vesta_ri.thresh,                 # p x q
            "ulvls_diag": vesta_ri.u_lvls,                  # d x (q x L)
            "masks": vesta_ri.masks,                        # d x L
        },
        "copse": {
            "answer": 1,
            "sc_length": len(copse_ri.lvls[0]),             # b
            "reshuf_colnum": len(copse_ri.reshuf),          # q
            "lvl_colnum": len(copse_ri.lvls[0]),            # b
            "facts_bits": copse_ri.facts,                   # p x q
            "thresh_bits": copse_ri.thresh,                 # p x q
            "reshuf_diag": copse_ri.reshuf,                 # q x q (diagonalized)
            "lvls_diag": copse_ri.lvls,                     # d x (b x L)
            "masks": copse_ri.masks,                        # d x L
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}.json")
    with open(out, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    nwin = sum(golden)
    print(f"  {name:<12} p={p} d={d} b={b} q={q} L={L}  winners={nwin}  -> {out}  ({os.path.getsize(out)//1024} KB)")

    if validate:
        _validate_against_openfhe(im, cm, inp, golden)
    return obj


def _validate_against_openfhe(im, cm, inp, golden):
    """Run the real OpenFHE IMM protocol and assert its decrypted result matches the golden."""
    from immed1ate.experiments.config import calculate_fhe_depth
    proto = IMMProtocol()
    depth = calculate_fhe_depth(cm.p, cm.d, "IMM", "OD")
    ri = proto.preprocessing(im, inp)
    scheme = proto.generate_fhe_scheme(im, depth, "BGV")
    ct = proto.encryption(scheme, ri)
    ctr = proto.evaluate(scheme, ct, cm.p)
    res = proto.decrypt(scheme, ctr, im)                 # list[bool]
    res01 = [1 if x else 0 for x in res]
    ok = (res01 == golden)
    print(f"    [validate] OpenFHE IMM vs golden: {'MATCH' if ok else 'MISMATCH'} "
          f"(winners fhe={sum(res01)} golden={sum(golden)})")
    if not ok:
        raise SystemExit(f"GOLDEN MISMATCH on {cm.name}")


def main():
    ap = argparse.ArgumentParser(description="Export 1MMED1ATE tensors to JSON for the SEAL backend")
    ap.add_argument("--model-folder", nargs="+", default=["data/models/rf"])
    ap.add_argument("--out-dir", default="data/exported")
    ap.add_argument("--validate", action="store_true",
                    help="run OpenFHE IMM on each model and assert the golden matches (slow on big models)")
    ap.add_argument("--only", default=None, help="comma-separated model stems to restrict to")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    mdls = []
    for folder in args.model_folder:
        mdls += sorted(glob.glob(os.path.join(folder, "*.mdl")))
    if only:
        mdls = [m for m in mdls if Path(m).stem in only]
    print(f"exporting {len(mdls)} models -> {args.out_dir}")
    for m in mdls:
        export_model(m, args.out_dir, validate=args.validate)
    print("DONE")


if __name__ == "__main__":
    main()
