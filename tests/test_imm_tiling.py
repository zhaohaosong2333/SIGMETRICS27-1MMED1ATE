#!/usr/bin/env python3
"""Regression test for 1MM's block-tiled MatMul (IMMProtocol with row_slots set).

When the 2q-wide edge vector's replicated copies do not fit one rotation row (N/2 slots), IMMProtocol
splits the edge columns into K blocks, each with its own SecComp and diagonal MatMul, and sums the
partial leaf counts. This test uses insecure toy rings so it runs in about a minute:
  * complete-tree model f7 (w = 1024 leaves, 2q = 2032 edge columns), p = 8, depth 5
  * N = 4096 (2048 slots per rotation row): untiled needs 2 x 2032 slots -> tiled into K = 3 blocks
  * N = 8192 (4096 slots): fits untiled
The decrypted leaf counts of both must equal the plaintext reference exactly.

    python tests/test_imm_tiling.py
"""
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from openfhe import CCParamsBGVRNS, GenCryptoContext, PKESchemeFeature, SecurityLevel
from immed1ate.compiler.extractor import get_unsecure_copse_model_from_serial_model as extract
from immed1ate.compiler.transformer import convert_copse_to_imm
from immed1ate.fhe.bgv import BGVScheme
from immed1ate.runtime.imm_protocol import IMMProtocol
from immed1ate.utils.characterize import reference_winners
from immed1ate.utils.encoding import generate_random_vector


def toy_scheme(ring, depth, rotations):
    params = CCParamsBGVRNS()
    params.SetPlaintextModulus(65537)
    params.SetMultiplicativeDepth(depth)
    params.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
    params.SetRingDim(ring)
    s = BGVScheme()
    s.crypto_context = GenCryptoContext(params)
    for feat in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE):
        s.crypto_context.Enable(feat)
    s.key_pair = s.crypto_context.KeyGen()
    s.crypto_context.EvalMultKeyGen(s.key_pair.secretKey)
    s.crypto_context.EvalRotateKeyGen(s.key_pair.secretKey, rotations)
    return s


def run(ring, cm, model, x):
    proto = IMMProtocol()
    proto.row_slots = ring // 2
    rt = proto.preprocessing(model, x)
    rotations = [-1, 1, -model.q * 2] + ([-proto.tile_width] if proto.tiles > 1 else [])
    scheme = toy_scheme(ring, model.p.bit_length() + 1, rotations)
    ct = proto.encryption(scheme, rt)
    res = proto.evaluate(scheme, ct, model.p)
    pt = scheme.decrypt(res)
    pt.SetLength(len(model.em))
    return proto.tiles, list(pt.GetPackedValue())


def main():
    cm = extract(str(REPO / "data/models/sweep/full/f7.mdl"))
    model = convert_copse_to_imm(cm)
    random.seed(20261001)
    x = generate_random_vector(cm.b, 2 ** cm.p - 1)
    _, ref_counts = reference_winners(cm, x)
    ok = True
    for ring in (4096, 8192):
        tiles, counts = run(ring, cm, model, x)
        same = counts == ref_counts
        ok &= same
        print(f"N={ring} ({ring // 2} slots/row): tiles={tiles}  leaf counts == plaintext reference: {same}")
    ok &= run(4096, cm, model, x)[0] > 1
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
