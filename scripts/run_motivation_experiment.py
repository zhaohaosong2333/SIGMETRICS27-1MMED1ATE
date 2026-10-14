#!/usr/bin/env python3
"""Motivation micro-benchmark: cost of ONE homomorphic matrix-vector multiply as a
function of the BGV multiplicative-depth budget.

Salvaged from the original 1MMED1ATE repo (Speed/Motivation/MatMulwithDifferentDepth_speed.py
and Memory/Motivation/MatMulwithDifferentDepth_mem.py). It shows FHE op cost -- both
time and peak memory -- rising with the multiplicative depth, with a step jump at the
ring-dimension boundary (~depth 8). That is the paper's motivation for 1MMED1ATE
compiling tree inference into a single, low multiplicative-depth matmul.

This is a deliberately standalone micro-benchmark of the raw packed matmul primitive
(its own FHEMatMul / diagonalization), decoupled from the tree pipeline, so it measures
the primitive cost the same way the original did. The correct diagonal-method
replication count is `1 + rownum // colnum` (for a square matrix, 2).

Modes:
  (default)          sweep depths [1..max_depth]; for each, run the matmul in a FRESH
                     subprocess and record encryption ms, compute ms, and peak RSS
                     (fresh process => clean peak, no FHE-context accumulation).
                     Writes a table to data/output/motivation_matmul_depth.txt.
  --single-depth D   run exactly one depth; print ENC_MS/COMP_MS/MEM_KB. Used
                     internally by the sweep; also handy for external profiling.

Example:
  python scripts/run_motivation_experiment.py                 # full 1024x1024, depths 1..15
  python scripts/run_motivation_experiment.py --rownum 256 --colnum 256 --max-depth 8
"""
import sys
import time
import random
import argparse
import subprocess
import resource
from pathlib import Path


def generate_random_matrix(rownum, colnum, n):
    return [[random.randint(0, n) for _ in range(colnum)] for _ in range(rownum)]


def generate_random_vector(length):
    return [random.randint(0, 1) for _ in range(length)]


def diagonalize_matrix(M):
    rownum = len(M)
    colnum = len(M[0])
    DM = []
    for result_row in range(colnum):
        DM.append([])
        for row in range(rownum):
            DM[result_row].append(M[row % rownum][(result_row + row) % colnum])
    return DM


def encrypt_matrix(cc, pk, M):
    return [cc.Encrypt(pk, cc.MakePackedPlaintext(vec)) for vec in M]


def encrypt_vector(cc, pk, v):
    return cc.Encrypt(pk, cc.MakePackedPlaintext(v))


def mv_mul(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def fhe_mat_mul(cc, ctxt_dm, ctxt_v, colnum, recursive_time=2):
    """Diagonal-method homomorphic matrix-vector multiply (faithful to the original)."""
    ctxt_mv = cc.EvalRotate(ctxt_v, -colnum)
    ctxt_ev = cc.EvalAdd(ctxt_mv, ctxt_v)
    for _ in range(recursive_time - 1):
        ctxt_mv = cc.EvalRotate(ctxt_mv, -colnum)
        ctxt_ev = cc.EvalAdd(ctxt_ev, ctxt_mv)

    rownum = len(ctxt_dm)
    ctxt_r = cc.EvalMult(ctxt_dm[0], ctxt_ev)
    for i in range(1, rownum):
        ctxt_ev = cc.EvalRotate(ctxt_ev, 1)
        ctxt_r = cc.EvalAdd(cc.EvalMult(ctxt_dm[i], ctxt_ev), ctxt_r)
    return ctxt_r


def run_one_depth(depth, rownum, colnum, domain, seed=0):
    """Build a fresh BGV context at `depth`, time enc + matmul, verify, return metrics."""
    from openfhe import CCParamsBGVRNS, GenCryptoContext, PKESchemeFeature

    random.seed(seed)
    M = generate_random_matrix(rownum, colnum, domain)
    v = generate_random_vector(colnum)
    recursive_time = 1 + (rownum // colnum)
    DM = diagonalize_matrix(M)
    F = mv_mul(M, v)

    params = CCParamsBGVRNS()
    params.SetPlaintextModulus(65537)
    params.SetMultiplicativeDepth(depth)
    cc = GenCryptoContext(params)
    for feat in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE):
        cc.Enable(feat)
    kp = cc.KeyGen()
    cc.EvalMultKeyGen(kp.secretKey)
    cc.EvalRotateKeyGen(kp.secretKey, [-1, 1, -colnum])

    t = time.time()
    ctxt_dm = encrypt_matrix(cc, kp.publicKey, DM)
    ctxt_v = encrypt_vector(cc, kp.publicKey, v)
    enc_ms = (time.time() - t) * 1000.0

    t = time.time()
    ctxt_r = fhe_mat_mul(cc, ctxt_dm, ctxt_v, colnum, recursive_time)
    comp_ms = (time.time() - t) * 1000.0

    ptxt_r = cc.Decrypt(ctxt_r, kp.secretKey)
    ptxt_r.SetLength(rownum)
    R = ptxt_r.GetPackedValue()
    assert all(R[i] == F[i] for i in range(len(F))), "FHE matmul result != plaintext"

    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: kilobytes
    return enc_ms, comp_ms, mem_kb


def parse_arguments():
    ap = argparse.ArgumentParser(description="Matmul-cost-vs-FHE-depth motivation benchmark")
    ap.add_argument("--rownum", type=int, default=1024)
    ap.add_argument("--colnum", type=int, default=1024)
    ap.add_argument("--domain", type=int, default=31)
    ap.add_argument("--max-depth", type=int, default=15)
    ap.add_argument("--single-depth", type=int, default=None,
                    help="Run exactly one depth and print ENC_MS/COMP_MS/MEM_KB (used by the sweep).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse_arguments()

    # Single-depth mode: one run, print metrics for the parent sweep to parse.
    if args.single_depth is not None:
        enc, comp, mem = run_one_depth(args.single_depth, args.rownum, args.colnum,
                                       args.domain, args.seed)
        print(f"ENC_MS {enc}")
        print(f"COMP_MS {comp}")
        print(f"MEM_KB {mem}")
        return

    out = args.out or str(Path(__file__).parent.parent / "data" / "output" / "motivation_matmul_depth.txt")
    depths = list(range(1, args.max_depth + 1))
    rows = []

    print("=" * 70)
    print("Motivation: matmul cost vs BGV multiplicative depth")
    print(f"matrix {args.rownum}x{args.colnum}, domain={args.domain}, recursive_time={1 + args.rownum // args.colnum}")
    print("=" * 70)
    print(f"{'depth':<7}{'enc_ms':<14}{'compute_ms':<14}{'peak_rss_mb':<12}")
    print("-" * 47)

    for d in depths:
        # Fresh subprocess per depth: clean peak RSS and no FHE-context accumulation.
        cp = subprocess.run(
            [sys.executable, __file__, "--single-depth", str(d),
             "--rownum", str(args.rownum), "--colnum", str(args.colnum),
             "--domain", str(args.domain), "--seed", str(args.seed)],
            capture_output=True, text=True)
        if cp.returncode != 0:
            last = cp.stderr.strip().splitlines()[-1] if cp.stderr.strip() else "unknown error"
            print(f"{d:<7}FAILED: {last}")
            rows.append((d, None, None, None))
            continue
        vals = {}
        for line in cp.stdout.splitlines():
            if line.startswith(("ENC_MS", "COMP_MS", "MEM_KB")):
                k, val = line.split()
                vals[k] = float(val)
        enc = vals.get("ENC_MS")
        comp = vals.get("COMP_MS")
        mem_mb = vals.get("MEM_KB", 0.0) / 1024.0
        rows.append((d, enc, comp, mem_mb))
        print(f"{d:<7}{enc:<14.1f}{comp:<14.1f}{mem_mb:<12.1f}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# Matmul cost vs BGV multiplicative depth "
                f"({args.rownum}x{args.colnum}, domain={args.domain})\n")
        f.write("# depth\tenc_ms\tcompute_ms\tpeak_rss_mb\n")
        for d, enc, comp, mem_mb in rows:
            if enc is None:
                f.write(f"{d}\tFAILED\n")
            else:
                f.write(f"{d}\t{enc:.1f}\t{comp:.1f}\t{mem_mb:.1f}\n")
        enc_list = [round(r[1], 3) for r in rows if r[1] is not None]
        comp_list = [round(r[2], 3) for r in rows if r[2] is not None]
        f.write("\n# original list format (encryption ms / computation ms):\n")
        f.write(str(enc_list) + "\n")
        f.write(str(comp_list) + "\n")

    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
