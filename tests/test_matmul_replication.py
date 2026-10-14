#!/usr/bin/env python3
"""Regression test for the diagonal MatMul's input replication (BGVScheme.mat_mul).

OpenFHE BGV packed rotations are cyclic over N/2 slots (two hypercube rows), not N. The original
mat_mul always builds 3 copies of the length-`cols` input vector, so whenever 3*cols > N/2 the third
copy wraps onto the first and the product is silently wrong. 1MM's edge vector is 2q wide, so this
bites 1MM first (MNIST-RF at OD: 3*2q = 11310 > 8192 at N=2^14).

`exact_replication=True` builds only ceil((rows+cols-1)/cols) copies (2 for every 1MM operand) and
raises if they do not fit. This test uses an insecure toy ring (N=4096 -> 2048 slots per rotation
row) so it runs in seconds:
  * rows=650, cols=700: legacy needs 3*700=2100 > 2048 slots -> WRONG; exact needs 2*700 -> correct
  * rows=1500, cols=700: needs 4 copies = 2800 > 2048 -> exact mode refuses instead of returning garbage
    (legacy would also be wrong here: 3 copies cannot cover rows > 2*cols + 1)

    python tests/test_matmul_replication.py
"""
import sys
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from openfhe import (CCParamsBGVRNS, GenCryptoContext, PKESchemeFeature, SecurityLevel)
from immed1ate.fhe.bgv import BGVScheme
from immed1ate.utils.matrix_ops import diagonalize_matrix


def toy_scheme(cols):
    params = CCParamsBGVRNS()
    params.SetPlaintextModulus(65537)
    params.SetMultiplicativeDepth(2)
    params.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
    params.SetRingDim(4096)
    s = BGVScheme()
    s.crypto_context = GenCryptoContext(params)
    for feat in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE):
        s.crypto_context.Enable(feat)
    s.key_pair = s.crypto_context.KeyGen()
    s.crypto_context.EvalMultKeyGen(s.key_pair.secretKey)
    s.crypto_context.EvalRotateKeyGen(s.key_pair.secretKey, [1, -cols])
    return s


def run(rows, cols, exact):
    rng = random.Random(rows * 31 + cols)
    m = [[rng.randint(0, 3) for _ in range(cols)] for _ in range(rows)]
    v = [rng.randint(0, 3) for _ in range(cols)]
    want = [sum(a * b for a, b in zip(row, v)) % 65537 for row in m]
    s = toy_scheme(cols)
    s.exact_replication = exact
    ct = s.mat_mul(s.encrypt_matrix(diagonalize_matrix(m)), s.encrypt_vector(v), cols)
    pt = s.decrypt(ct)
    pt.SetLength(rows)
    return list(pt.GetPackedValue()) == want


def main():
    ok = True
    legacy = run(650, 700, exact=False)
    exact = run(650, 700, exact=True)
    print(f"rows=650 cols=700  legacy 3 copies (2100 > 2048 slots): correct={legacy}  (expected False)")
    print(f"rows=650 cols=700  exact 2 copies  (1400 <= 2048 slots): correct={exact}   (expected True)")
    ok &= (not legacy) and exact
    try:
        run(1500, 700, exact=True)
        print("rows=1500 cols=700 exact: no error raised (expected a capacity error)")
        ok = False
    except ValueError as e:
        print(f"rows=1500 cols=700 exact: refused as expected -> {e}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
