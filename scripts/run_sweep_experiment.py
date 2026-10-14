#!/usr/bin/env python3
"""
Sensitivity / characterization sweep over model characteristics and FHE parameters.

For every model of a sweep family (scripts/gen_sweep_models.py) and every protocol
(IMM = 1MMED1ATE, VESTA, COPSE) this runs ONE isolated inference per process under GNU
`time -v`, exactly like the paper's memory harness (run_complete_experiment.py), and records:

  runtime    encryption / evaluation (SecComp, MatMul, level ops) / decryption, same stage
             boundaries as ExperimentRunner._run_protocol; plus compile-side matrix
             construction (parse + IR, IMM/VESTA transform, bit-decomposition + diagonalization)
  memory     peak RSS of the process (GNU time, the methodology of Table 9) + VmRSS/VmHWM at every
             stage boundary (which stage sets the peak)
  FHE params multiplicative depth, ring dimension N (and whether depth or slot capacity set it),
             log2 Q, bytes per ciphertext, evaluation-key bytes
  volume     ciphertexts per party (model owner / client), encrypted-model bytes
  op counts  every OpenFHE call counted + timed per stage (rotations, ct-ct mults, adds, ...)
  correct    decrypted result vs a plaintext evaluation of the same forest on the same input

Settings: OD / UD (the paper's two depth configurations), or DEPTH: each protocol at every
multiplicative depth from its own OD minimum to --depth-max (FHE-parameter sweep).

Two runtime options are ON by default:
  * exact MatMul replication (2 copies of the 1MM input vector instead of 3; capacity-checked)
  * slot capacity: if a MatMul does not fit N/2 rotation slots at the depth's ring, 1MM tiles its
    MatMul into column blocks (same ring and depth); the baselines would get a larger ring
--legacy-matmul restores the 3-copy / depth-only behaviour and --no-imm-tiling the ring-raising
behaviour, for A/B comparison.

    python scripts/run_sweep_experiment.py --model-folder data/models/sweep/ens --config-setting OD \
        --output-folder reproduce/_run/sw_ens_od
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PROTOCOLS = ["IMM", "VESTA", "COPSE"]


# ============================================================================================
# Worker: one (model, protocol, depth) inference, prints one JSON line
# ============================================================================================
def run_worker(args):
    import gc
    import random
    from immed1ate.utils import characterize as C

    mem = {"start": C.mem_snapshot()}
    rec = {"model": Path(args.model).stem, "protocol": args.protocol, "setting": args.setting,
           "legacy_matmul": bool(args.legacy_matmul), "seed": args.seed}

    # ---- compile (model owner, plaintext): parse -> COPSE IR -> protocol representation ----
    from immed1ate.compiler.extractor import get_unsecure_copse_model_from_serial_model as extract
    from immed1ate.compiler.transformer import convert_copse_to_imm, convert_copse_to_vesta
    t0 = time.perf_counter()
    cm = extract(args.model)
    rec["t_extract_ms"] = (time.perf_counter() - t0) * 1e3
    p, d, b, q, w = cm.p, cm.d, cm.b, cm.q, len(cm.masks[0])
    rec.update(p=p, d=d, b=b, q=q, w=w)
    t0 = time.perf_counter()
    if args.protocol == "IMM":
        model = convert_copse_to_imm(cm)
    elif args.protocol == "VESTA":
        model = convert_copse_to_vesta(cm)
    else:
        model = cm
    rec["t_transform_ms"] = (time.perf_counter() - t0) * 1e3
    mem["compiled"] = C.mem_snapshot()

    from immed1ate.experiments.config import calculate_fhe_depth
    from immed1ate.utils.encoding import generate_random_vector
    from immed1ate.runtime.imm_protocol import IMMProtocol
    from immed1ate.runtime.vesta_protocol import VESTAProtocol
    from immed1ate.runtime.copse_protocol import COPSEProtocol
    import immed1ate.fhe.bgv as bgv
    from immed1ate.utils.matrix_ops import plan_column_tiling
    import openfhe

    random.seed(args.seed)
    x = generate_random_vector(cm.b, 2 ** p - 1)
    protocol = {"IMM": IMMProtocol, "VESTA": VESTAProtocol, "COPSE": COPSEProtocol}[args.protocol]()

    # ---- FHE parameters: depth -> OpenFHE's 128-bit ring. If a MatMul does not fit one rotation row
    # (N/2 slots) there, 1MM splits its edge columns into blocks (IMMProtocol tiling) and keeps the ring;
    # otherwise (baselines, --no-imm-tiling, or a split that cannot fit) the ring is raised. ----
    depth = args.depth if args.depth > 0 else calculate_fhe_depth(p, d, args.protocol, args.setting)
    n_sec = bgv.BGVScheme.ring_dimension(depth)
    need = C.required_row_slots(args.protocol, p, q, b, d, w, legacy=args.legacy_matmul)
    tiles, tile_width = 1, None
    if (args.protocol == "IMM" and not args.legacy_matmul and not args.no_imm_tiling
            and need > n_sec // 2):
        plan = plan_column_tiling(w, 2 * q, n_sec // 2)
        if plan:
            tiles, tile_width = plan
            protocol.row_slots = n_sec // 2
            need = C.replication_copies(w, tile_width) * tile_width
    ring = n_sec
    if not args.legacy_matmul:
        while ring // 2 < need:
            ring *= 2
    rec.update(fhe_depth=depth, ring_dim_security=n_sec, ring_dim=ring, row_slots_needed=need,
               ring_bound="slots" if ring > n_sec else "depth",
               slot_overflow=bool(need > ring // 2), imm_tiles=tiles, imm_tile_width=tile_width)

    t0 = time.perf_counter()
    runtime_input = protocol.preprocessing(model, x)
    rec["t_preprocess_ms"] = (time.perf_counter() - t0) * 1e3
    mem["preprocessed"] = C.mem_snapshot()
    if args.protocol == "IMM":
        assert getattr(runtime_input, "tiles", 1) == tiles, "IMM tiling plan mismatch"

    orig_generate = bgv.BGVScheme.generate_scheme

    def generate_with_ring(self, depth, rotation_list, plaintext_modulus=65537, ring_dim=None):
        return orig_generate(self, depth, rotation_list, plaintext_modulus,
                             ring_dim=ring if ring != n_sec else None)
    bgv.BGVScheme.generate_scheme = generate_with_ring

    t0 = time.perf_counter()
    scheme = protocol.generate_fhe_scheme(model, depth, "BGV")
    rec["t_keygen_ms"] = (time.perf_counter() - t0) * 1e3
    scheme.exact_replication = not args.legacy_matmul
    cc = scheme.crypto_context
    rec["ring_dim_actual"] = cc.GetRingDimension()
    rec["log2_Q"] = float(int(cc.GetModulus()).bit_length())
    mem["keygen"] = C.mem_snapshot()

    counter = C.OpCounter(cc)
    scheme.crypto_context = counter
    matmul_log = []
    C.label_stages(scheme, counter, matmul_log)

    # ---- timed inference: identical stage boundaries to ExperimentRunner._run_protocol ----
    counter.stage = "Encrypt"
    t0 = time.perf_counter()
    ct_input = protocol.encryption(scheme, runtime_input)
    rec["encryption"] = (time.perf_counter() - t0) * 1e3
    mem["encrypted"] = C.mem_snapshot()

    counter.stage = "Eval"
    t0 = time.perf_counter()
    ct_result, eval_timing = protocol.evaluate(scheme, ct_input, p, return_timing=True)
    rec["evaluation"] = (time.perf_counter() - t0) * 1e3
    rec.update({k: v for k, v in eval_timing.items()})
    mem["evaluated"] = C.mem_snapshot()

    counter.stage = "Decrypt"
    t0 = time.perf_counter()
    result = protocol.decrypt(scheme, ct_result, model)
    rec["decryption"] = (time.perf_counter() - t0) * 1e3
    rec["total"] = rec["encryption"] + rec["evaluation"] + rec["decryption"]
    mem["decrypted"] = C.mem_snapshot()

    # ---- after the timed run: correctness, volume, key sizes (cannot raise the peak) ----
    counter.stage = "Verify"
    pt = cc.Decrypt(ct_result, scheme.key_pair.secretKey)
    pt.SetLength(w)
    raw = list(pt.GetPackedValue())
    ref_winners, ref_counts = C.reference_winners(cm, x)
    got = [1 if v else 0 for v in result]
    rec["n_trees_won"] = sum(ref_winners)
    rec["correct"] = got == ref_winners
    rec["mismatch_leaves"] = sum(1 for a, c in zip(got, ref_winners) if a != c)
    if args.protocol == "IMM":
        rec["imm_counts_exact"] = raw == ref_counts

    n_model = n_client = 0
    for name, val in vars(ct_input).items():
        stack, n = [val], 0
        while stack:
            v = stack.pop()
            if isinstance(v, list):
                stack.extend(v)
            elif not isinstance(v, (int, float, str)):
                n += 1
        if "fact" in name:
            n_client += n
        else:
            n_model += n
    rec.update(ct_model=n_model, ct_client=n_client)
    fresh = cc.Encrypt(scheme.key_pair.publicKey, cc.MakePackedPlaintext([0]))
    rec["ct_bytes"] = len(openfhe.Serialize(fresh, openfhe.BINARY))
    rec["bytes_model_ct"] = n_model * rec["ct_bytes"]
    rec["bytes_client_ct"] = n_client * rec["ct_bytes"]
    del ct_input, ct_result, runtime_input, fresh
    gc.collect()
    rec["bytes_key_mult"] = len(openfhe.SerializeEvalMultKeyString(openfhe.BINARY, ""))
    rec["bytes_key_rot"] = len(openfhe.SerializeEvalAutomorphismKeyString(openfhe.BINARY, ""))

    rec.update(counter.flat())
    rec["matmul_calls"] = matmul_log
    rec["mem_kb"] = mem
    print("@@RESULT " + json.dumps(rec, default=float), flush=True)


# ============================================================================================
# Driver: loop models x protocols x depths, one GNU-time'd subprocess each
# ============================================================================================
def gnu_time():
    cand = os.environ.get("GNU_TIME") or "/usr/bin/time"
    if (os.path.isabs(cand) and os.path.exists(cand)) or shutil.which(cand):
        return cand
    return None


def parse_time_v(stderr: str):
    rss_kb = wall = None
    for line in stderr.splitlines():
        line = line.strip()
        if line.startswith("Maximum resident set size"):
            rss_kb = int(line.split()[-1])
        elif line.startswith("Elapsed (wall clock) time"):
            parts = line.split()[-1].split(":")
            wall = sum(float(x) * 60 ** i for i, x in enumerate(reversed(parts)))
    return rss_kb, wall


def plan_runs(models, protocols, setting, depth_min, depth_max):
    from immed1ate.compiler.extractor import get_unsecure_copse_model_from_serial_model as extract
    from immed1ate.experiments.config import calculate_fhe_depth
    runs = []
    for mdl in models:
        if setting in ("OD", "UD"):
            runs += [(mdl, proto, setting, -1) for proto in protocols]
            continue
        cm = extract(str(mdl))
        for proto in protocols:
            lo = max(depth_min, calculate_fhe_depth(cm.p, cm.d, proto, "OD"))
            runs += [(mdl, proto, "DEPTH", L) for L in range(lo, depth_max + 1)]
    return runs


def run_driver(args):
    folder = Path(args.model_folder)
    models = sorted(folder.glob("*.mdl"))
    if args.models:
        models = [m for m in models if m.stem in set(args.models)]
    if not models:
        sys.exit(f"no models in {folder}")
    out = Path(args.output_folder)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = Path(args.jsonl) if args.jsonl else out / "runs.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and jsonl.exists():
        for line in jsonl.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add((r["model"], r["protocol"], r["setting"], r.get("fhe_depth_req", -1)))
    tm = gnu_time()
    if tm is None:
        print("WARNING: no GNU time -v; peak RSS will be missing", file=sys.stderr)

    runs = plan_runs(models, args.protocols, args.config_setting, args.depth_min, args.depth_max)
    runs = [r for r in runs if (r[0].stem, r[1], r[2], r[3]) not in done]
    print(f"== sweep {folder} setting={args.config_setting} runs={len(runs)} -> {jsonl}", flush=True)
    for i, (mdl, proto, setting, depth) in enumerate(runs, 1):
        cmd = [sys.executable, __file__, "--worker", "--model", str(mdl), "--protocol", proto,
               "--setting", setting if setting != "DEPTH" else "OD", "--depth", str(depth),
               "--seed", str(args.seed)]
        if args.legacy_matmul:
            cmd.append("--legacy-matmul")
        if args.no_imm_tiling:
            cmd.append("--no-imm-tiling")
        if tm:
            cmd = [tm, "-v"] + cmd
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        rec = None
        for line in proc.stdout.splitlines():
            if line.startswith("@@RESULT "):
                rec = json.loads(line[len("@@RESULT "):])
        rss_kb, wall = parse_time_v(proc.stderr)
        if rec is None:
            tail = (proc.stderr or proc.stdout)[-3000:]
            rec = {"model": mdl.stem, "protocol": proto, "error": tail}
        rec.update(family=folder.name, setting=setting, returncode=proc.returncode,
                   peak_rss_kb=rss_kb, wall_s=wall if wall is not None else time.time() - t0)
        if setting == "DEPTH":
            rec["fhe_depth_req"] = depth
        (out / "logs").mkdir(exist_ok=True)
        (out / "logs" / f"{mdl.stem}__{proto}__{setting}{depth if depth > 0 else ''}.log").write_text(
            proc.stdout + "\n----- stderr -----\n" + proc.stderr)
        with open(jsonl, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        if "error" in rec:
            print(f"[{i}/{len(runs)}] {mdl.stem:6s} {proto:5s} {setting} L={depth}: FAILED rc={proc.returncode}",
                  flush=True)
        else:
            print(f"[{i}/{len(runs)}] {mdl.stem:6s} {proto:5s} {setting:5s} L={rec['fhe_depth']:2d} "
                  f"N={rec['ring_dim']:5d}({rec['ring_bound']}) total={rec['total'] / 1e3:8.1f}s "
                  f"peak={(rss_kb or 0) / 2 ** 20:6.2f}GB correct={rec['correct']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-folder")
    ap.add_argument("--models", nargs="+", help="subset of model names (stems) to run")
    ap.add_argument("--config-setting", default="OD", choices=["OD", "UD", "DEPTH"])
    ap.add_argument("--depth-min", type=int, default=0)
    ap.add_argument("--depth-max", type=int, default=12)
    ap.add_argument("--protocols", nargs="+", default=PROTOCOLS, choices=PROTOCOLS)
    ap.add_argument("--output-folder", default="reproduce/_run/sweep", help="per-run worker logs")
    ap.add_argument("--jsonl", help="results file (default <output-folder>/runs.jsonl); appended to")
    ap.add_argument("--resume", action="store_true", help="skip runs already completed in --jsonl")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--legacy-matmul", action="store_true")
    ap.add_argument("--no-imm-tiling", action="store_true",
                    help="raise the ring instead of tiling 1MM's MatMul when it does not fit N/2 slots")
    # worker mode (internal)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", help=argparse.SUPPRESS)
    ap.add_argument("--protocol", help=argparse.SUPPRESS)
    ap.add_argument("--setting", help=argparse.SUPPRESS)
    ap.add_argument("--depth", type=int, default=-1, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        run_worker(args)
    else:
        if not args.model_folder:
            ap.error("--model-folder is required")
        run_driver(args)


if __name__ == "__main__":
    main()
