"""Measurement helpers for the model-characteristics / FHE-parameter sweep
(scripts/gen_sweep_models.py, scripts/run_sweep_experiment.py).

Nothing here changes what the IMM / VESTA / COPSE protocols compute. It provides:
  * static_profile()      hardware-independent structure + compiled-representation profile of a
                          .mdl (tree shape, density, padding, ciphertext counts, element counts)
  * matmul_shapes() /     the (rows x cols) of every homomorphic MatMul a protocol performs, and
    required_row_slots()  the rotation-row slots its diagonal MatMul needs (sets the ring floor)
  * OpCounter             transparent CryptoContext proxy that counts and times every OpenFHE call
  * mem_snapshot()        VmRSS / VmHWM of the current process
  * reference_winners()   plaintext tree evaluation used to verify every decrypted FHE result

Lightweight on purpose: importable without OpenFHE (static profiling runs anywhere).
"""

import math
import time
from collections import defaultdict
from typing import Dict, List, Tuple

PROTOCOLS = ("IMM", "VESTA", "COPSE")


# --------------------------------------------------------------------------------------------
# Compiled-representation shapes
# --------------------------------------------------------------------------------------------
def matmul_shapes(protocol: str, p: int, q: int, b: int, d: int, w: int) -> List[Tuple[str, int, int, int]]:
    """(operand, rows, cols, count) of every encrypted matrix a protocol multiplies by a vector.

    Diagonal encoding turns a rows x cols matrix into `cols` ciphertexts of `rows` slots, and the
    MatMul costs `cols` rotations + `cols` ct-ct multiplications.
      IMM   em      : w x 2q            (one MatMul)
      VESTA ulvl[l] : w x q   for l < d (d MatMuls)
      COPSE reshuf  : b x q   (one)  +  lvl[l] : w x b  for l < d (d MatMuls)
    """
    if protocol == "IMM":
        return [("em", w, 2 * q, 1)]
    if protocol == "VESTA":
        return [("ulvl", w, q, d)]
    if protocol == "COPSE":
        return [("reshuf", b, q, 1), ("lvl", w, b, d)]
    raise ValueError(protocol)


def replication_copies(rows: int, cols: int) -> int:
    """Copies of the length-`cols` input vector the diagonal MatMul needs so that slot
    i + j (i < rows, j < cols) holds v[(i + j) mod cols] (BGVScheme.mat_mul builds >= 2)."""
    return max(2, math.ceil((rows + cols - 1) / cols))


def required_row_slots(protocol: str, p: int, q: int, b: int, d: int, w: int,
                       legacy: bool = False) -> int:
    """Slots one rotation row must hold for every MatMul of `protocol` to be exact.

    OpenFHE BGV/BFV packed rotations are cyclic over N/2 slots (two hypercube rows), so a
    ring dimension N is only valid if this is <= N/2. `legacy` reproduces the original
    BGVScheme.mat_mul, which always builds 3 copies regardless of the shape.
    """
    need = 0
    for _, rows, cols, _ in matmul_shapes(protocol, p, q, b, d, w):
        copies = 3 if legacy else replication_copies(rows, cols)
        need = max(need, copies * cols, rows)
    return need


def ciphertext_counts(protocol: str, p: int, q: int, b: int, d: int, w: int) -> Dict[str, int]:
    """Ciphertexts per party, matching the protocols' encryption() exactly."""
    if protocol == "IMM":
        model = p + 2 * q              # dual-threshold bit-planes + em diagonals
    elif protocol == "VESTA":
        model = p + d * q + d          # threshold bit-planes + d ulvl matrices + d masks
    else:
        model = p + q + d * b + d      # threshold bit-planes + reshuf + d lvl matrices + d masks
    return {"ct_model": model, "ct_client": p, "ct_total": model + p}


# --------------------------------------------------------------------------------------------
# Static (hardware-independent) profile of one serialized model
# --------------------------------------------------------------------------------------------
def _leaf_depths(tree) -> List[int]:
    root = next(br for br in tree.branch_dict.values() if br.level == 0)
    out, stack = [], [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if node.is_leaf():
            out.append(depth)
        else:
            stack.append((node.left, depth + 1))
            stack.append((node.right, depth + 1))
    return out


def static_profile(path: str) -> Dict:
    """Tree structure, compiled sizes and padding of one .mdl for all three systems.

    Padding is reported at three layers, each of which the paper's 9.1 analysis touches:
      level padding  : leaves shallower than the forest depth d are padded to d levels
                       (COPSE/VESTA masks + level matrices; 1MM duplicated edge weights)
      matrix padding : structural zeros of the dense operand (encrypted like any other entry)
      slot padding   : unused slots of each diagonal ciphertext (rows / (N/2), reported by the
                       FHE run because N depends on depth)
    """
    import numpy as np
    from ..compiler.extractor import (get_copse_model_from_serial_file, unsecure_copse_model,
                                      read_forest_from_file)
    from ..compiler.transformer import convert_copse_to_imm, convert_copse_to_vesta

    rf = read_forest_from_file(path)
    trees = [rf.tree_dict[name] for name in rf.get_tree_names()]
    leaf_depths = [_leaf_depths(t) for t in trees]
    tree_depths = [max(ld) for ld in leaf_depths]
    n_branch = [len(t.branch_dict) for t in trees]
    all_leaf_depths = [x for ld in leaf_depths for x in ld]

    t0 = time.perf_counter()
    raw = get_copse_model_from_serial_file(path)
    t_extract = time.perf_counter() - t0
    q_slots = len(raw.thresh)                                  # per-feature padded threshold slots
    used = [any(row[j] for row in raw.reshuf) for j in range(q_slots)]
    zero_thresh = sum(1 for j in range(q_slots) if used[j] and raw.thresh[j] == 0)
    k = raw.k
    cm = unsecure_copse_model(raw)                             # what every runtime actually uses
    p, q, b, d = cm.p, cm.q, cm.b, cm.d
    w = len(cm.masks[0])

    t0 = time.perf_counter()
    im = convert_copse_to_imm(cm)
    t_imm = time.perf_counter() - t0
    t0 = time.perf_counter()
    vm = convert_copse_to_vesta(cm)
    t_vesta = time.perf_counter() - t0

    em = np.asarray(im.em, dtype=np.int16)
    ul = np.asarray(vm.ulvls, dtype=np.int8)
    lv = np.asarray(cm.lvls, dtype=np.int8)
    rs = np.asarray(cm.reshuf, dtype=np.int8)

    prof = {
        "model": rf.name, "p": p, "d": d, "b": b, "q": q, "w": w, "k": k,
        "n_trees": len(trees),
        "tree_depth_min": min(tree_depths), "tree_depth_mean": float(np.mean(tree_depths)),
        "leaf_depth_mean": float(np.mean(all_leaf_depths)),
        "branches_per_tree_mean": float(np.mean(n_branch)),
        # density = branch nodes / branch nodes of complete trees of the same per-tree depths
        "density": sum(n_branch) / sum(2 ** td - 1 for td in tree_depths),
        "q_slots_padded": q_slots, "feature_pad_slots": q_slots - sum(used),
        "zero_thresh": zero_thresh,
        "level_pad_entries": sum(d - x for x in all_leaf_depths),
        "level_pad_frac": sum(d - x for x in all_leaf_depths) / (d * w),
        "edges": 2 * b,
        "path_edges_total": sum(all_leaf_depths),
        "compile_extract_s": t_extract, "compile_imm_s": t_imm, "compile_vesta_s": t_vesta,
        # dense element counts of every structure operand (what is diagonalized + encrypted)
        "elem_IMM": int(em.size), "nnz_IMM": int(np.count_nonzero(em)),
        "elem_VESTA": int(ul.size), "nnz_VESTA": int(np.count_nonzero(ul)),
        "elem_COPSE": int(lv.size + rs.size), "nnz_COPSE": int(np.count_nonzero(lv) + np.count_nonzero(rs)),
        "em_max_weight": int(em.max()) if em.size else 0,
    }
    for proto in PROTOCOLS:
        prof[f"zero_frac_{proto}"] = 1.0 - prof[f"nnz_{proto}"] / max(1, prof[f"elem_{proto}"])
        cts = ciphertext_counts(proto, p, q, b, d, w)
        prof[f"ct_model_{proto}"] = cts["ct_model"]
        prof[f"ct_total_{proto}"] = cts["ct_total"]
        prof[f"matmul_rot_{proto}"] = sum(cols * cnt for _, _, cols, cnt in matmul_shapes(proto, p, q, b, d, w))
        prof[f"row_slots_{proto}"] = required_row_slots(proto, p, q, b, d, w)
        prof[f"row_slots_legacy_{proto}"] = required_row_slots(proto, p, q, b, d, w, legacy=True)
    return prof


# --------------------------------------------------------------------------------------------
# Runtime instrumentation
# --------------------------------------------------------------------------------------------
TRACKED_OPS = ("EvalMult", "EvalRotate", "EvalAdd", "EvalSub", "EvalNegate",
               "MakePackedPlaintext", "Encrypt", "Decrypt")


class OpCounter:
    """Transparent proxy around an OpenFHE CryptoContext: counts and times each tracked call,
    attributed to the current `stage` label. Per-call overhead is ~1 us against ms-scale ops."""

    def __init__(self, cc):
        object.__setattr__(self, "_cc", cc)
        object.__setattr__(self, "stage", "setup")
        object.__setattr__(self, "stats", defaultdict(lambda: [0, 0.0]))

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        attr = getattr(self._cc, name)
        if name not in TRACKED_OPS:
            return attr
        stats = self.stats

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            out = attr(*args, **kwargs)
            rec = stats[(self.stage, name)]
            rec[0] += 1
            rec[1] += time.perf_counter() - t0
            return out
        return timed

    def flat(self) -> Dict[str, float]:
        out = {}
        for (stage, op), (cnt, sec) in sorted(self.stats.items()):
            out[f"op_{stage}_{op}_n"] = cnt
            out[f"op_{stage}_{op}_ms"] = sec * 1000.0
        return out


def label_stages(scheme, counter: OpCounter, matmul_log: List[Dict]):
    """Wrap the scheme's evaluation primitives so every OpenFHE call is tagged with the
    protocol stage it belongs to (SecComp / MatMul / LevelOp)."""
    def wrap(method_name, stage):
        orig = getattr(scheme, method_name)

        def labelled(*args, **kwargs):
            prev = counter.stage
            counter.stage = stage
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                if stage == "MatMul":
                    ctxt_matrix, colnum = args[0], args[2]
                    matmul_log.append({"cols": colnum, "rows": getattr(ctxt_matrix, "slot_len", None),
                                       "ms": (time.perf_counter() - t0) * 1000.0})
                counter.stage = prev
        setattr(scheme, method_name, labelled)

    wrap("sec_comp_8bit", "SecComp")
    wrap("sec_comp_16bit", "SecComp")
    wrap("mat_mul", "MatMul")
    wrap("eval_add", "LevelOp")    # COPSE/VESTA per-level mask add
    wrap("eval_mult", "LevelOp")   # COPSE/VESTA cross-level reduction


def mem_snapshot() -> Dict[str, int]:
    """Current and peak resident set of this process (kB) from /proc/self/status."""
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, val = line.split(":")
                    out[key] = int(val.split()[0])
    except OSError:
        pass
    return out


# --------------------------------------------------------------------------------------------
# Plaintext reference (ground truth for every decrypted result)
# --------------------------------------------------------------------------------------------
def reference_winners(cm, x: List[int]) -> Tuple[List[int], List[int]]:
    """Plaintext evaluation of the compiled forest on runtime input `x`.

    Mirrors the runtime semantics: slot j compares x[j] < thresh[j]; reshuf routes slot
    decisions to branches; for leaf r at level l the selected branch is lvls[l][r]'s one-hot
    column and the edge is valid iff (mask=0 and decision=1) or (mask=1 and decision=0).

    Returns (winners, valid_edge_counts): winners[r] = 1 iff every level's edge is valid, and
    valid_edge_counts[r] = number of valid levels (what a correct 1MM decrypts; == d for winners).
    """
    dec = [1 if x[j] < t else 0 for j, t in enumerate(cm.thresh)]
    bdec = []
    for row in cm.reshuf:
        try:
            bdec.append(dec[row.index(1)])
        except ValueError:            # branch whose threshold column was stripped (thresh == 0)
            bdec.append(0)
    w = len(cm.masks[0])
    counts = [0] * w
    for lvl, mask in zip(cm.lvls, cm.masks):
        for r in range(w):
            br = lvl[r].index(1)
            counts[r] += 1 if (bdec[br] == 1) != (mask[r] == 1) else 0
    winners = [1 if c == cm.d else 0 for c in counts]
    return winners, counts
