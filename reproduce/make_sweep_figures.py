#!/usr/bin/env python3
"""Figures, tables and fitted numbers of the sensitivity study (the sw_* tags of reproduce/manifest.tsv).

Outputs (default reproduce/figures_paper/):
  sweep_scaling.pdf        time + peak memory vs ensemble size / per-tree size / depth (fixed b) /
                           complete-tree depth, COPSE / VESTA / 1MM at OD and UD
  sweep_costmodel.pdf      the 9.1 cost model against every run: MatMul time vs #diagonal products,
                           encryption time vs #ciphertexts, peak RSS vs ciphertext + key bytes
  sweep_fheparam.pdf       multiplicative depth L (-> ring N, log Q): per-op latency, end-to-end
                           time and peak memory per protocol
  sweep_model.pdf          paper figure: per-op latency vs L, time vs L, MatMul time vs #diagonals
  sweep_membreakdown.pdf   what the peak memory is made of (keys / encrypted model / client / rest)
  tab_sweep_summary.md     per model: time, speedup, peak memory, memory ratio at OD and UD
  tab_sweep_precision.md   precision x size
  tab_sweep_fheparam.md    time / memory per protocol at every multiplicative depth L
  tab_sweep_static.md      compiled representation: elements, zero fraction, ciphertexts, slots,
                           matrix-construction time
  tab_sweep_volume.md      data volume + capacity (encrypted model, keys, client upload)
  sweep_fits.txt           scaling exponents, cost-model fits, regimes, correctness, cross-check

    python reproduce/make_sweep_figures.py [--results data/paper_results] [--outdir reproduce/figures_paper]
"""
import argparse
import csv
import glob
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

import figstyle as fs

PLAIN = FuncFormatter(lambda v, _: f"{v:g}")


def plain_log_axes(ax, x=False):
    """log axes with 1 / 10 / 100 style labels instead of 10^k / 4x10^0."""
    ax.yaxis.set_major_formatter(PLAIN)
    ax.yaxis.set_minor_formatter(NullFormatter())
    if x:
        ax.xaxis.set_major_formatter(PLAIN)
        ax.xaxis.set_minor_formatter(NullFormatter())

ROOT = Path(__file__).resolve().parent.parent
HJ = ROOT / "data/paper_results"   # results directory; set by --results
PROTOS = ["COPSE", "VESTA", "IMM"]
LABEL = {"COPSE": "COPSE", "VESTA": "VESTA", "IMM": "1MM"}
NUM = ["fhe_depth", "ring_dim", "log2_Q", "p", "d", "b", "q", "w", "total", "encryption",
       "evaluation", "decryption", "SecComp", "MatMul", "Reordering", "LvlProcessing", "Reduction",
       "t_extract_ms", "t_transform_ms", "t_preprocess_ms", "t_keygen_ms", "peak_rss_kb",
       "rss_preprocessed_kb", "ct_model", "ct_client", "ct_bytes", "bytes_model_ct",
       "bytes_client_ct", "bytes_key_mult", "bytes_key_rot", "eval_rot_n", "eval_rot_ms",
       "eval_mult_n", "eval_mult_ms", "matmul_rot_n", "matmul_mult_n", "matmul_ms", "encrypt_n",
       "encode_ms", "encrypt_ms", "n_trees", "density", "level_pad_frac", "zero_frac_IMM",
       "zero_frac_VESTA", "zero_frac_COPSE", "elem_IMM", "elem_VESTA", "elem_COPSE", "row_slots_needed"]
GB = 2 ** 30


# ------------------------------------------------------------------------------------------
# loading + rep aggregation
# ------------------------------------------------------------------------------------------
def tcrit(n):
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    return table.get(n, 1.96)


def load():
    files = sorted(glob.glob(str(HJ / "sw_*__*__rep*.csv")))
    files = [f for f in files if not Path(f).name.startswith("sw_smoke")]
    raw = []
    for f in files:
        rep = int(re.search(r"__rep(\d+)", f).group(1))
        for r in csv.DictReader(open(f)):
            if r.get("error") or r.get("correct") != "True":
                continue
            for k in NUM:
                try:
                    r[k] = float(r[k]) if r.get(k, "") != "" else float("nan")
                except ValueError:
                    r[k] = float("nan")
            r["rep"] = rep
            raw.append(r)
    groups = defaultdict(list)
    for r in raw:
        groups[(r["family"], r["model"], r["protocol"], r["setting"], int(r["fhe_depth"]))].append(r)
    agg = []
    for key, rs in groups.items():
        a = dict(rs[0])
        a["reps"] = len(rs)
        for k in ("total", "encryption", "evaluation", "peak_rss_kb", "matmul_ms"):
            vals = [x[k] for x in rs]
            a[k] = statistics.mean(vals)
            a[k + "_ci"] = (tcrit(len(vals)) * statistics.stdev(vals) / math.sqrt(len(vals))
                            if len(vals) > 1 else 0.0)
        agg.append(a)
    return raw, agg, files


def index(agg):
    return {(a["family"], a["model"], a["protocol"], a["setting"]): a for a in agg if a["setting"] in ("OD", "UD")}


def fam_models(agg, fam):
    rows = {}
    for a in agg:
        if a["family"] == fam:
            rows[a["model"]] = a
    return sorted(rows, key=lambda m: (rows[m]["b"], rows[m]["d"], m))


# ------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------
B_REF = 248      # depth family: halving the trees per level lets b drift 224->255; plot at this b
PANELS = [("ens", "b", "ensemble size\n(4-128 trees, d=6)", "#branches b"),
          ("size", "b", "tree size\n(8 trees, d=8)", "#branches b"),
          ("depth", "d", f"depth, scaled to b={B_REF}\n(32 to 1 trees)", "depth d"),
          ("full", "b", "complete trees\n(8 trees, d=3-8)", "#branches b")]
# second tick-label line for the b axes whose points also move another parameter
TICK2 = {"full": lambda a: f"d={int(a['d'])}"}          # b = 8 * (2^d - 1)


def fig_scaling(agg, out):
    idx = index(agg)
    fig, axes = fs.new_fig(fs.COL_W, 3.5, nrows=2, ncols=4)
    for col, (fam, xkey, title, xlabel) in enumerate(PANELS):
        models = fam_models(agg, fam)
        for row, (ykey, scale, ylabel) in enumerate((("total", 1e-3, "time (s)"),
                                                     ("peak_rss_kb", 1024 / 1e9, "peak mem. (GB)"))):
            ax = axes[row][col]

            def xy(a):           # time and memory are linear in b (exponents 0.97-1.00, ens family)
                return a[xkey], a[ykey] * scale * (B_REF / a["b"] if fam == "depth" else 1)
            for proto in PROTOS:
                for setting, ls in (("OD", "-"), ("UD", "--")):
                    pts = [xy(idx[(fam, m, proto, setting)]) for m in models if (fam, m, proto, setting) in idx]
                    if not pts:
                        continue
                    xs, ys = zip(*sorted(pts))
                    ax.plot(xs, ys, ls, color=fs.PROTO_COLOR[proto], marker="o", markersize=2.5,
                            linewidth=1.0)
                    slot = [xy(idx[(fam, m, proto, setting)]) for m in models if (fam, m, proto, setting) in idx
                            and idx[(fam, m, proto, setting)].get("ring_bound") == "slots"]
                    for sx, sy in slot:      # ring raised by the MatMul slot requirement, not by depth
                        ax.plot([sx], [sy], marker="o", markersize=6, markerfacecolor="none",
                                markeredgecolor=fs.INK, markeredgewidth=0.7, ls="")
            ax.set_yscale("log")
            if xkey == "b":
                ax.set_xscale("log", base=2)
            plain_log_axes(ax, x=(xkey == "b"))
            if xkey == "d":          # pin ticks: the auto locator thins them when panels are narrow
                ax.set_xticks(sorted({int(idx[k]["d"]) for k in idx if k[0] == fam}))
            elif fam not in TICK2:
                ax.xaxis.set_major_locator(LogLocator(base=4, subs=[1.0]))
            if fam in TICK2:         # label every other point, ending at the largest
                pts = sorted((idx[(fam, m, "IMM", "OD")] for m in models if (fam, m, "IMM", "OD") in idx),
                             key=lambda a: a["b"])
                ticks = pts[(len(pts) - 1) % 2::2]
                ax.set_xticks([int(a["b"]) for a in ticks],
                              [f"{int(a['b'])}\n{TICK2[fam](a)}" for a in ticks])
                ax.set_xticks([int(a["b"]) for a in pts], minor=True)
                ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which="major", axis="y")
            if row == 0:
                ax.set_title(title, fontsize=7.5)
            else:
                ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.tick_params(labelsize=7)
    handles = fs.line_handles([(LABEL[p], fs.PROTO_COLOR[p], "o") for p in PROTOS])
    from matplotlib.lines import Line2D
    handles += [Line2D([0], [0], color=fs.INK, ls="-", lw=1.0, label="OD"),
                Line2D([0], [0], color=fs.INK, ls="--", lw=1.0, label="UD")]
    if any(a.get("ring_bound") == "slots" for a in agg):
        handles.append(Line2D([0], [0], marker="o", markersize=6, markerfacecolor="none",
                              markeredgecolor=fs.INK, markeredgewidth=0.7, ls="", label="N raised by slots"))
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
    fig.align_xlabels(axes[1])
    fs.save(fig, out / "sweep_scaling.pdf")
    print(f"  saved sweep_scaling.pdf")


def loglog_fit(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    slope, icpt = np.polyfit(x, y, 1)
    pred = slope * x + icpt
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return slope, math.exp(icpt), r2


def fig_costmodel(raw, out, fits):
    """Every run (all families, OD/UD/DEPTH): does the 9.1 operation count explain it?"""
    fig, axes = fs.new_fig(fs.COL_W, 2.0, ncols=3)
    markers = {16384: "o", 32768: "s", 65536: "^"}
    specs = [("matmul_mult_n", "matmul_ms", 1e-3, "#diagonal products", "MVMul time (s)"),
             ("encrypt_n", "encryption", 1e-3, "#ciphertexts encrypted", "encryption time (s)"),
             ("ct_bytes_total", "peak_rss_kb", 1 / 2 ** 20, "ciphertext + key bytes (GB)", "peak RSS (GB)")]
    for r in raw:
        r["ct_bytes_total"] = ((r["ct_model"] + r["ct_client"]) * r["ct_bytes"]
                               + r["bytes_key_mult"] + r["bytes_key_rot"]) / GB
    for ax, (xk, yk, ys, xl, yl) in zip(axes, specs):
        for proto in PROTOS:
            for ring, mk in markers.items():
                pts = [(r[xk], r[yk] * ys) for r in raw if r["protocol"] == proto and r["ring_dim"] == ring
                       and r[xk] > 0 and r[yk] > 0]
                if pts:
                    xs, yv = zip(*pts)
                    ax.scatter(xs, yv, s=7, marker=mk, color=fs.PROTO_COLOR[proto], linewidths=0,
                               alpha=0.85)
        ax.set_xscale("log")
        ax.set_yscale("log")
        plain_log_axes(ax, x=True)
        ax.set_xlabel(xl, fontsize=7)
        ax.set_ylabel(yl, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, which="major")
        for ring in markers:
            pts = [(r[xk], r[yk] * ys) for r in raw if r["ring_dim"] == ring and r[xk] > 0 and r[yk] > 0]
            if len(pts) >= 3:
                slope, coef, r2 = loglog_fit(*zip(*pts))
                fits.append(f"cost model  {yk:>12s} ~ {xk:<14s} N={ring:<6d} slope={slope:5.3f}  "
                            f"coef={coef:.4g}  R2={r2:.4f}  (n={len(pts)})")
        if xk == "ct_bytes_total":
            lo = min(r[xk] for r in raw)
            hi = max(r[xk] for r in raw)
            ax.plot([lo, hi], [lo, hi], color=fs.INK, lw=0.6, ls=":")
    handles = fs.line_handles([(LABEL[p], fs.PROTO_COLOR[p], "o") for p in PROTOS])
    from matplotlib.lines import Line2D
    handles += [Line2D([0], [0], color=fs.INK, marker=mk, ls="", markersize=4, label=f"N=2$^{{{int(math.log2(n))}}}$")
                for n, mk in markers.items() if any(r["ring_dim"] == n for r in raw)]
    fig.legend(handles=handles, loc="outside upper center", ncol=6)
    fs.save(fig, out / "sweep_costmodel.pdf")
    print("  saved sweep_costmodel.pdf")


def fig_fheparam(raw, agg, out, fits):
    rows = [a for a in agg if a["family"] == "fhe" and a["setting"] == "DEPTH"]
    if not rows:
        print("  [skip] sweep_fheparam.pdf: no sw_fhe data")
        return
    fig, axes = fs.new_fig(fs.COL_W, 2.0, ncols=3)
    # (a) per-op latency vs L: protocol/model independent -> pooled over all fhe runs
    per = defaultdict(lambda: defaultdict(list))
    for r in raw:
        if r["family"] == "fhe" and r["eval_rot_n"] > 0:
            per["rotation"][int(r["fhe_depth"])].append(r["eval_rot_ms"] / r["eval_rot_n"])
            per["ct$\\times$ct mult"][int(r["fhe_depth"])].append(r["eval_mult_ms"] / r["eval_mult_n"])
            per["encrypt"][int(r["fhe_depth"])].append(r["encrypt_ms"] / max(1, r["encrypt_n"]))
    ax = axes[0]
    for (name, data), color in zip(per.items(), (fs.TOL["indigo"], fs.TOL["wine"], fs.TOL["sand"])):
        Ls = sorted(data)
        ax.plot(Ls, [statistics.median(data[L]) for L in Ls], "-o", color=color, markersize=2.5,
                lw=1.0, label=name)
        fits.append(f"per-op ms   {name:<20s} " + "  ".join(f"L{L}={statistics.median(data[L]):.2f}"
                                                         for L in Ls))
    ring_by_L = {int(r["fhe_depth"]): int(r["ring_dim"]) for r in raw if r["family"] == "fhe"}
    jump = [L for L in sorted(ring_by_L) if ring_by_L[L] > ring_by_L.get(L - 1, ring_by_L[L])]
    for ax_ in axes:
        for L in jump:
            ax_.axvline(L - 0.5, color=fs.GRID, lw=0.8)
    ax.set_ylabel("ms per op", fontsize=7)
    ax.legend(fontsize=6, loc="upper left")
    for ax_, (key, scale, yl) in zip(axes[1:], (("total", 1e-3, "end-to-end time (s)"),
                                                 ("peak_rss_kb", 1 / 2 ** 20, "peak memory (GB)"))):
        for model, ls in (("n064", "-"), ("n016", ":")):
            for proto in PROTOS:
                pts = sorted((a["fhe_depth"], a[key] * scale) for a in rows
                             if a["model"] == model and a["protocol"] == proto)
                if pts:
                    xs, ys = zip(*pts)
                    ax_.plot(xs, ys, ls, color=fs.PROTO_COLOR[proto], marker="o", markersize=2.5, lw=1.0)
        ax_.set_yscale("log")
        plain_log_axes(ax_)
        ax_.set_ylabel(yl, fontsize=7)
    for ax_ in axes:
        ax_.set_xlabel("multiplicative depth L", fontsize=7)
        ax_.tick_params(labelsize=7)
        ax_.grid(True, which="major", axis="y")
    from matplotlib.lines import Line2D
    handles = fs.line_handles([(LABEL[p], fs.PROTO_COLOR[p], "o") for p in PROTOS])
    handles += [Line2D([0], [0], color=fs.INK, ls="-", lw=1.0, label="b=1984"),
                Line2D([0], [0], color=fs.INK, ls=":", lw=1.0, label="b=496")]
    fig.legend(handles=handles, loc="outside upper center", ncol=5)
    fs.save(fig, out / "sweep_fheparam.pdf")
    print("  saved sweep_fheparam.pdf")


def fig_paper_model(raw, agg, out):
    """Paper figure (one row): (a) per-operation latency vs multiplicative depth L, (b) end-to-end
    time vs L for the b=1984 forest, (c) MatMul time vs #diagonal ct x ct products over every run."""
    rows = [a for a in agg if a["family"] == "fhe" and a["setting"] == "DEPTH"]
    if not rows:
        print("  [skip] sweep_model.pdf: no sw_fhe data")
        return
    fig, axes = fs.new_fig(fs.COL_W, 1.9, ncols=3)
    per = defaultdict(lambda: defaultdict(list))
    for r in raw:
        if r["family"] == "fhe" and r["eval_rot_n"] > 0:
            per["rotation"][int(r["fhe_depth"])].append(r["eval_rot_ms"] / r["eval_rot_n"])
            per["ct$\\times$ct mult."][int(r["fhe_depth"])].append(r["eval_mult_ms"] / r["eval_mult_n"])
    ring_by_L = {int(r["fhe_depth"]): int(r["ring_dim"]) for r in raw if r["family"] == "fhe"}
    jump = [L for L in sorted(ring_by_L) if ring_by_L[L] > ring_by_L.get(L - 1, ring_by_L[L])]
    ax = axes[0]
    for (name, data), color, mk in zip(per.items(), (fs.TOL["indigo"], fs.TOL["wine"]), ("o", "s")):
        Ls = sorted(data)
        ax.plot(Ls, [statistics.median(data[L]) for L in Ls], "-", marker=mk, color=color,
                markersize=2.5, lw=1.0, label=name)
    ax.set_ylabel("ms / operation", fontsize=7)
    ax.legend(fontsize=6, loc="lower right", handlelength=1.2)
    ax = axes[1]
    for proto in PROTOS:
        pts = sorted((a["fhe_depth"], a["total"] / 1e3) for a in rows
                     if a["model"] == "n064" and a["protocol"] == proto)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-o", color=fs.PROTO_COLOR[proto], markersize=2.5, lw=1.0)
    ax.set_yscale("log")
    plain_log_axes(ax)
    from matplotlib.ticker import LogLocator
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.set_ylabel("time (s), b=1984", fontsize=7)
    for ax_ in axes[:2]:
        for L in jump:
            ax_.axvline(L - 0.5, color=fs.INK, lw=0.5, ls=":")
            ax_.text(L - 0.55, ax_.get_ylim()[1], f"N=2$^{{{int(math.log2(ring_by_L[L - 1]))}}}$ | ",
                     fontsize=6, va="top", ha="right")
            ax_.text(L - 0.45, ax_.get_ylim()[1], f"N=2$^{{{int(math.log2(ring_by_L[L]))}}}$",
                     fontsize=6, va="top", ha="left")
        ax_.set_xlabel("multiplicative depth $L$", fontsize=7)
        ax_.tick_params(labelsize=7)
        ax_.grid(True, which="major", axis="y")
    ax = axes[2]
    markers = {16384: "o", 32768: "s"}
    for proto in PROTOS:
        for ring, mk in markers.items():
            pts = [(r["matmul_mult_n"], r["matmul_ms"] / 1e3) for r in raw
                   if r["protocol"] == proto and r["ring_dim"] == ring and r["matmul_mult_n"] > 0]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, s=6, marker=mk, color=fs.PROTO_COLOR[proto], linewidths=0, alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_axes(ax, x=True)
    ax.set_xlabel("#diagonal products", fontsize=7)
    ax.set_ylabel("MVMul time (s)", fontsize=7)
    for ax_, tag in zip(axes, "abc"):
        ax_.set_title(f"({tag})", fontsize=7, loc="left")
    ax.tick_params(labelsize=7)
    ax.grid(True, which="major")
    from matplotlib.lines import Line2D
    handles = fs.line_handles([(LABEL[p], fs.PROTO_COLOR[p], "o") for p in PROTOS])
    handles += [Line2D([0], [0], color=fs.INK, marker=mk, ls="", markersize=3.5,
                       label=f"N=2$^{{{int(math.log2(n))}}}$") for n, mk in markers.items()]
    fig.legend(handles=handles, loc="outside upper center", ncol=5)
    fs.save(fig, out / "sweep_model.pdf")
    print("  saved sweep_model.pdf")


def memory_shares(raw, fits):
    """How much of the peak RSS is ciphertext (encrypted model + client) + evaluation keys."""
    shares = defaultdict(list)
    for r in raw:
        if r["peak_rss_kb"] > 0 and r["peak_rss_kb"] * 1024 > 1e9:      # runs above 1 GB (decimal)
            ct = (r["bytes_model_ct"] + r["bytes_client_ct"] + r["bytes_key_mult"] + r["bytes_key_rot"]) / 1024
            shares[r["protocol"]].append(ct / r["peak_rss_kb"])
    for proto in PROTOS:
        if shares[proto]:
            v = shares[proto]
            fits.append(f"ciphertext+key share of peak RSS (runs > 1 GB) {LABEL[proto]:<5s}: "
                        f"median={100 * statistics.median(v):.1f}% min={100 * min(v):.1f}% max={100 * max(v):.1f}%")


def fig_membreakdown(agg, out):
    idx = index(agg)
    picks = [("ens", "n064"), ("full", "f8")]
    bars = []
    for fam, model in picks:
        for setting in ("OD", "UD"):
            for proto in PROTOS:
                a = idx.get((fam, model, proto, setting))
                if a:
                    bars.append((f"{model} {setting} {LABEL[proto]}", a))
    if not bars:
        print("  [skip] sweep_membreakdown.pdf: no ens/n064 or full/f8 data")
        return
    comps = [("eval. keys", lambda a: (a["bytes_key_mult"] + a["bytes_key_rot"]) / GB, fs.TOL["indigo"]),
             ("encrypted model", lambda a: a["bytes_model_ct"] / GB, fs.TOL["rose"]),
             ("client ciphertexts", lambda a: a["bytes_client_ct"] / GB, fs.TOL["sand"]),
             ("plaintext model / IR", lambda a: a["rss_preprocessed_kb"] / 2 ** 20, fs.TOL["cyan"]),
             ("evaluation temporaries + other", None, fs.TOL["olive"])]
    fig, ax = fs.new_fig(fs.COL_W, 0.22 * len(bars) + 0.7)
    for i, (label, a) in enumerate(bars):
        left = 0.0
        vals = [f(a) if f else None for _, f, _ in comps]
        vals[-1] = max(0.0, a["peak_rss_kb"] / 2 ** 20 - sum(v for v in vals[:-1]))
        for (name, _, color), v in zip(comps, vals):
            ax.barh(i, v, left=left, color=color, edgecolor=fs.INK, linewidth=0.4, height=0.7,
                    label=name if i == 0 else None)
            left += v
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b[0] for b in bars], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("peak RSS (GB)", fontsize=7)
    ax.tick_params(labelsize=7)
    fig.legend(loc="outside upper center", ncol=3, fontsize=6.5)
    fs.save(fig, out / "sweep_membreakdown.pdf")
    print("  saved sweep_membreakdown.pdf")


# ------------------------------------------------------------------------------------------
# tables + fitted numbers
# ------------------------------------------------------------------------------------------
def fmt_t(a):
    s = a["total"] / 1e3
    ci = a.get("total_ci", 0) / 1e3
    return f"{s:.1f}" + (f"±{ci:.1f}" if ci > 0 else "")


def tab_summary(agg, out, fits):
    idx = index(agg)
    lines = ["# Extended sensitivity sweep: end-to-end time and peak memory (Xeon Gold 6444Y, 32 threads)\n",
             "Time = encryption + evaluation + decryption (s); speedup = baseline / 1MM; mem = 1MM peak RSS "
             "as % of the baseline's.\n"]
    for fam in ("ens", "size", "depth", "full", "prec"):
        models = fam_models(agg, fam)
        if not models:
            continue
        lines += [f"\n## {fam}\n",
                  "| model | n | d | b | density | set | COPSE s | VESTA s | 1MM s | ×COPSE | ×VESTA | "
                  "COPSE GB | VESTA GB | 1MM GB | mem %COPSE | mem %VESTA | 1MM N |",
                  "|---|--:|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for m in models:
            for setting in ("OD", "UD"):
                c, v, i = (idx.get((fam, m, p, setting)) for p in PROTOS)
                if not i:
                    continue
                g = lambda a: a["peak_rss_kb"] / 2 ** 20 if a else float("nan")
                sc = c["total"] / i["total"] if c else float("nan")
                sv = v["total"] / i["total"] if v else float("nan")
                lines.append(
                    f"| {m} | {int(i['n_trees'])} | {int(i['d'])} | {int(i['b'])} | {i['density']:.2f} | {setting} | "
                    f"{fmt_t(c) if c else '–'} | {fmt_t(v) if v else '–'} | {fmt_t(i)} | {sc:.2f} | {sv:.2f} | "
                    f"{g(c):.1f} | {g(v):.1f} | {g(i):.1f} | {100 * g(i) / g(c):.1f} | {100 * g(i) / g(v):.1f} | "
                    f"2^{int(math.log2(i['ring_dim']))}{'*' if i.get('ring_bound') == 'slots' else ''} |")
        # regime numbers per family
        for setting in ("OD", "UD"):
            sp = [(m, idx[(fam, m, 'COPSE', setting)]["total"] / idx[(fam, m, 'IMM', setting)]["total"])
                  for m in models if (fam, m, "COPSE", setting) in idx and (fam, m, "IMM", setting) in idx]
            if sp:
                fits.append(f"speedup vs COPSE  {fam:<6s} {setting}: " +
                            "  ".join(f"{m}={s:.2f}" for m, s in sp))
            sv = [(m, idx[(fam, m, 'VESTA', setting)]["total"] / idx[(fam, m, 'IMM', setting)]["total"])
                  for m in models if (fam, m, "VESTA", setting) in idx and (fam, m, "IMM", setting) in idx]
            if sv:
                fits.append(f"speedup vs VESTA  {fam:<6s} {setting}: " +
                            "  ".join(f"{m}={s:.2f}" for m, s in sv))
            mr = [(m, idx[(fam, m, 'IMM', setting)]["peak_rss_kb"] / idx[(fam, m, 'COPSE', setting)]["peak_rss_kb"])
                  for m in models if (fam, m, "COPSE", setting) in idx and (fam, m, "IMM", setting) in idx]
            if mr:
                fits.append(f"mem 1MM/COPSE    {fam:<6s} {setting}: " +
                            "  ".join(f"{m}={100 * s:.1f}%" for m, s in mr))
    lines.append("\n`*` ring dimension raised by the MatMul's slot requirement rather than by the depth "
                 "(1MM's 2q-wide edge vector needs 4q rotation slots).\n")
    (out / "tab_sweep_summary.md").write_text("\n".join(lines) + "\n")
    print("  saved tab_sweep_summary.md")


def scaling_exponents(agg, fits):
    idx = index(agg)
    for fam, xkey in (("ens", "b"), ("size", "b"), ("full", "b")):
        for setting in ("OD", "UD"):
            for proto in PROTOS:
                for ykey, name in (("total", "time"), ("peak_rss_kb", "mem")):
                    rows = [idx[(fam, m, proto, setting)] for m in fam_models(agg, fam)
                            if (fam, m, proto, setting) in idx]
                    if not rows:
                        continue
                    same = [a for a in rows if a["ring_dim"] == rows[0]["ring_dim"]]
                    other = [a["model"] for a in rows if a["ring_dim"] != rows[0]["ring_dim"]]
                    if len(same) >= 3:
                        slope, _, r2 = loglog_fit([a[xkey] for a in same], [a[ykey] for a in same])
                        fits.append(f"exponent {name:<4s} ~ {xkey}^a  {fam:<5s} {setting} {LABEL[proto]:<5s} "
                                    f"a={slope:5.2f}  R2={r2:.3f}  (N=2^{int(math.log2(same[0]['ring_dim']))}"
                                    + (f"; excluded ring change: {','.join(other)})" if other else ")"))
    for setting in ("OD", "UD"):
        for proto in PROTOS:
            rows = sorted((idx[("depth", m, proto, setting)] for m in fam_models(agg, "depth")
                           if ("depth", m, proto, setting) in idx), key=lambda a: a["d"])
            if len(rows) >= 3:
                lo, hi = rows[0], rows[-1]
                fits.append(f"depth d={int(lo['d'])}->{int(hi['d'])} at b={int(lo['b'])}->{int(hi['b'])}: "
                            f"time x{hi['total'] / lo['total']:.2f}  time/b x{(hi['total'] / hi['b']) / (lo['total'] / lo['b']):.2f}  "
                            f"mem x{hi['peak_rss_kb'] / lo['peak_rss_kb']:.2f}  {setting} {LABEL[proto]}")


def tab_precision(agg, out):
    idx = index(agg)
    models = fam_models(agg, "prec")
    if not models:
        return
    lines = ["# Precision p in {8,16} x ensemble size (Income RF, d=6)\n",
             "| model | p | b | set | L (1MM/VESTA/COPSE) | COPSE s | VESTA s | 1MM s | SecComp share of 1MM eval | ×COPSE | 1MM GB | COPSE GB |",
             "|---|--:|--:|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for m in models:
        for setting in ("OD", "UD"):
            c, v, i = (idx.get((("prec"), m, p, setting)) for p in PROTOS)
            if not (c and v and i):
                continue
            lines.append(f"| {m} | {int(i['p'])} | {int(i['b'])} | {setting} | "
                         f"{int(i['fhe_depth'])}/{int(v['fhe_depth'])}/{int(c['fhe_depth'])} | {fmt_t(c)} | "
                         f"{fmt_t(v)} | {fmt_t(i)} | {100 * i['SecComp'] / i['evaluation']:.0f}% | "
                         f"{c['total'] / i['total']:.2f} | {i['peak_rss_kb'] / 2 ** 20:.1f} | "
                         f"{c['peak_rss_kb'] / 2 ** 20:.1f} |")
    (out / "tab_sweep_precision.md").write_text("\n".join(lines) + "\n")
    print("  saved tab_sweep_precision.md")


def tab_fheparam(agg, out, fits):
    rows = [a for a in agg if a["family"] == "fhe" and a["setting"] == "DEPTH"]
    if not rows:
        return
    Ls = sorted({int(a["fhe_depth"]) for a in rows})
    get = {(a["model"], a["protocol"], int(a["fhe_depth"])): a for a in rows}
    lines = ["# FHE-parameter sweep: end-to-end time (s) / peak memory (GB) vs multiplicative depth L\n",
             "| model | protocol | " + " | ".join(f"L={L}" for L in Ls) + " |",
             "|---|---|" + "--:|" * len(Ls)]
    for model in sorted({a["model"] for a in rows}):
        for proto in PROTOS:
            cells = []
            for L in Ls:
                a = get.get((model, proto, L))
                cells.append(f"{a['total'] / 1e3:.1f} / {a['peak_rss_kb'] / 2 ** 20:.1f}" if a else "–")
            b = next(int(a["b"]) for a in rows if a["model"] == model)
            lines.append(f"| {model} (b={b}) | {LABEL[proto]} | " + " | ".join(cells) + " |")
        for L in Ls:
            c, i = get.get((model, "COPSE", L)), get.get((model, "IMM", L))
            v = get.get((model, "VESTA", L))
            if c and i and v:
                fits.append(f"fixed-L speedup {model} L={L}: vs COPSE {c['total'] / i['total']:.2f}  "
                            f"vs VESTA {v['total'] / i['total']:.2f}")
    (out / "tab_sweep_fheparam.md").write_text("\n".join(lines) + "\n")
    print("  saved tab_sweep_fheparam.md")


def tab_static(agg, out):
    """Compiled representation (hardware independent) + measured matrix-construction time."""
    idx = index(agg)
    lines = ["# Compiled representation per model (static) and matrix-construction time (6444Y)\n",
             "elements = dense entries of the encrypted structure operands; zero % = structural padding "
             "that is encrypted anyway; ct = ciphertexts the model owner encrypts; slots = rotation-row "
             "slots the MatMul needs (must be <= N/2); build = parse + IR + transform + diagonalize (s).\n",
             "| family | model | b | d | w | lvl-pad % | 1MM elem | VESTA elem | COPSE elem | 1MM zero % | "
             "1MM ct | VESTA ct | COPSE ct | 1MM slots | build 1MM s | build COPSE s |",
             "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for f in sorted(glob.glob(str(ROOT / "data/sweep/static/*.csv"))):
        fam = Path(f).stem
        for r in csv.DictReader(open(f)):
            def build(proto):
                a = idx.get((fam, r["model"], proto, "OD"))
                if not a:
                    return "–"
                return f"{(a['t_extract_ms'] + a['t_transform_ms'] + a['t_preprocess_ms']) / 1e3:.2f}"
            lines.append(
                f"| {fam} | {r['model']} | {r['b']} | {r['d']} | {r['w']} | {100 * float(r['level_pad_frac']):.1f} | "
                f"{int(r['elem_IMM']):,} | {int(r['elem_VESTA']):,} | {int(r['elem_COPSE']):,} | "
                f"{100 * float(r['zero_frac_IMM']):.2f} | {int(r['ct_model_IMM']):,} | {int(r['ct_model_VESTA']):,} | "
                f"{int(r['ct_model_COPSE']):,} | {int(r['row_slots_IMM']):,} | {build('IMM')} | {build('COPSE')} |")
    (out / "tab_sweep_static.md").write_text("\n".join(lines) + "\n")
    print("  saved tab_sweep_static.md")


def tab_volume(agg, out, fits):
    idx = index(agg)
    lines = ["# Data volume and capacity (bytes measured by serializing the actual ciphertexts / keys; "
             "GB = 2^30 B, MB = 2^20 B like the peak-RSS columns; GPU80 compares against 80e9 B)\n",
             "model→server = encrypted model (one-time per model); client→server = encrypted query bits; "
             "keys = relinearization + rotation keys. `GPU80` = encrypted model + keys fit in 80 GB.\n",
             "MatMul GB = encrypted matrix diagonals streamed once per inference (#diagonals × bytes/ct); "
             "MatMul GB/s = that volume / MatMul time on the 6444Y (a lower bound on the DRAM traffic).\n",
             "| family | model | set | proto | N | log2 Q | MB / ct | model→server GB | client→server MB | keys MB | MatMul GB | MatMul GB/s | peak GB | GPU80 |",
             "|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
    rates = []
    for fam in ("ens", "full"):
        for m in fam_models(agg, fam):
            for setting in ("OD", "UD"):
                for proto in PROTOS:
                    a = idx.get((fam, m, proto, setting))
                    if not a:
                        continue
                    work = (a["bytes_model_ct"] + a["bytes_key_mult"] + a["bytes_key_rot"]) / 1e9
                    mm_gb = a["matmul_mult_n"] * a["ct_bytes"] / GB
                    rate = mm_gb / (a["matmul_ms"] / 1e3)
                    rates.append(rate)
                    lines.append(f"| {fam} | {m} | {setting} | {LABEL[proto]} | 2^{int(math.log2(a['ring_dim']))} | "
                                 f"{int(a['log2_Q'])} | {a['ct_bytes'] / 2 ** 20:.2f} | {a['bytes_model_ct'] / GB:.2f} | "
                                 f"{a['bytes_client_ct'] / 2 ** 20:.1f} | {(a['bytes_key_mult'] + a['bytes_key_rot']) / 2 ** 20:.0f} | "
                                 f"{mm_gb:.1f} | {rate:.2f} | "
                                 f"{a['peak_rss_kb'] / 2 ** 20:.1f} | {'yes' if work <= 80 else 'no'} |")
    if rates:
        fits.append(f"MatMul operand streaming rate (GB/s, 6444Y): min={min(rates):.2f} "
                    f"median={statistics.median(rates):.2f} max={max(rates):.2f}")
    (out / "tab_sweep_volume.md").write_text("\n".join(lines) + "\n")
    print("  saved tab_sweep_volume.md")


def predictive_check(agg, fits):
    """Calibrate per-operation costs on the SMALLEST model of a family (same protocol, setting,
    ring N and depth L), then predict every larger model from its operation counts alone:
        T = enc/ct * #ct  +  SecComp  +  MatMul/diagonal * #diagonals  +  rest
    Peak memory is predicted the same way from ciphertext bytes. Reports |error| per point."""
    idx = index(agg)
    for fam in ("ens", "size", "full", "depth"):
        models = fam_models(agg, fam)
        for setting in ("OD", "UD"):
            for proto in PROTOS:
                pts = [idx[(fam, m, proto, setting)] for m in models if (fam, m, proto, setting) in idx]
                if len(pts) < 3:
                    continue
                cal = pts[0]
                errs, mem_errs, skipped = [], [], []
                for a in pts[1:]:
                    if a["ring_dim"] != cal["ring_dim"] or a["fhe_depth"] != cal["fhe_depth"]:
                        skipped.append(f"{a['model']}(N=2^{int(math.log2(a['ring_dim']))},L={int(a['fhe_depth'])})")
                        continue
                    rest = cal["evaluation"] - cal["SecComp"] - cal["matmul_ms"]
                    pred = (cal["encryption"] / cal["encrypt_n"] * a["encrypt_n"] + cal["SecComp"]
                            + cal["matmul_ms"] / cal["matmul_mult_n"] * a["matmul_mult_n"]
                            + rest * a["d"] / cal["d"] + cal["decryption"])
                    errs.append(abs(pred - a["total"]) / a["total"])
                    ct = lambda r: (r["ct_model"] + r["ct_client"]) * r["ct_bytes"] / 1024
                    base = cal["peak_rss_kb"] - ct(cal)
                    mem_errs.append(abs(base + ct(a) - a["peak_rss_kb"]) / a["peak_rss_kb"])
                if errs:
                    fits.append(f"predict from {cal['model']:<5s} {fam:<5s} {setting} {LABEL[proto]:<5s} "
                                f"time |err| median={100 * statistics.median(errs):4.1f}% max={100 * max(errs):4.1f}%  "
                                f"mem |err| median={100 * statistics.median(mem_errs):4.1f}% max={100 * max(mem_errs):4.1f}%"
                                + (f"  (other N or L, not predicted: {', '.join(skipped)})" if skipped else ""))


def op_decomposition(agg, fits):
    """Split the OD speedup into (i) operation-count ratio, visible at UD where all systems share
    N and L, and (ii) the FHE-parameter gain of 1MM's shallower circuit (its UD/OD time ratio)."""
    idx = index(agg)
    for fam in ("ens", "size", "full", "depth", "prec"):
        for m in fam_models(agg, fam):
            k = lambda p, s: idx.get((fam, m, p, s))
            if not all(k(p, s) for p in ("COPSE", "IMM") for s in ("OD", "UD")):
                continue
            ud = k("COPSE", "UD")["total"] / k("IMM", "UD")["total"]
            param = k("IMM", "UD")["total"] / k("IMM", "OD")["total"]
            base = k("COPSE", "UD")["total"] / k("COPSE", "OD")["total"]
            od = k("COPSE", "OD")["total"] / k("IMM", "OD")["total"]
            diag = k("COPSE", "UD")["matmul_mult_n"] / k("IMM", "UD")["matmul_mult_n"]
            fits.append(f"OD speedup decomposition {fam:<5s} {m:<8s} d={int(k('IMM', 'OD')['d']):2d}: "
                        f"#diag ratio={diag:4.2f}  UD speedup={ud:4.2f}  x 1MM param gain={param:4.2f}  "
                        f"/ COPSE param gain={base:4.2f}  = OD speedup {od:5.2f}")


def equal_bd_pairs(agg, fits):
    """Same b and d, different tree count / density: is cost a function of (b, d) only?"""
    idx = index(agg)
    for (fa, ma), (fb, mb) in ((("full", "f6"), ("ens", "n016")), (("full", "f5"), ("depth", "d05")),
                               (("full", "f8"), ("size", "l256"))):
        for setting in ("OD", "UD"):
            parts = []
            for proto in PROTOS:
                a, b = idx.get((fa, ma, proto, setting)), idx.get((fb, mb, proto, setting))
                if a and b:
                    parts.append(f"{LABEL[proto]} time/b {(a['total'] / a['b']) / (b['total'] / b['b']):.3f} "
                                 f"mem/b {(a['peak_rss_kb'] / a['b']) / (b['peak_rss_kb'] / b['b']):.3f}")
            if parts:
                A, B = idx[(fa, ma, "IMM", setting)], idx[(fb, mb, "IMM", setting)]
                fits.append(f"equal-(b,d) {ma}(n={int(A['n_trees'])},b={int(A['b'])},d={int(A['d'])},dens={A['density']:.2f}) vs "
                            f"{mb}(n={int(B['n_trees'])},b={int(B['b'])},d={int(B['d'])},dens={B['density']:.2f}) {setting}: "
                            + "  ".join(parts))


def main():
    global HJ
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "reproduce/figures_paper"))
    ap.add_argument("--results", default=str(HJ), help="results directory with the sw_*.csv (default data/paper_results)")
    args = ap.parse_args()
    HJ = Path(args.results)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    fs.use_style()
    raw, agg, files = load()
    if not raw:
        print(f"  [skip] sweep figures: no sw_* CSVs in {HJ}")
        return
    fits = [f"# sweep fits ({len(files)} CSVs, {len(raw)} correct runs, {len(agg)} configurations)"]
    allruns = []
    for f in files:
        allruns += list(csv.DictReader(open(f)))
    bad = [r for r in allruns if r.get("correct") != "True"]
    fits.append(f"correctness: {len(allruns) - len(bad)}/{len(allruns)} runs decrypted to the plaintext "
                f"reference" + ("" if not bad else "  FAILED: " + ", ".join(
                    f"{r['family']}/{r['model']}/{r['protocol']}/{r['setting']}" for r in bad)))
    fig_scaling(agg, out)
    fig_costmodel(raw, out, fits)
    fig_fheparam(raw, agg, out, fits)
    fig_paper_model(raw, agg, out)
    fig_membreakdown(agg, out)
    tab_summary(agg, out, fits)
    scaling_exponents(agg, fits)
    memory_shares(raw, fits)
    for a in agg:
        if a["protocol"] == "IMM" and float(a.get("imm_tiles") or 1) > 1:
            fits.append(f"1MM MatMul tiled: {a['family']}/{a['model']} {a['setting']} into {int(float(a['imm_tiles']))} "
                        f"column blocks of {int(float(a['imm_tile_width']))} (N=2^{int(math.log2(a['ring_dim']))} kept)")
    tab_precision(agg, out)
    tab_fheparam(agg, out, fits)
    tab_static(agg, out)
    tab_volume(agg, out, fits)
    predictive_check(agg, fits)
    op_decomposition(agg, fits)
    equal_bd_pairs(agg, fits)
    (out / "sweep_fits.txt").write_text("\n".join(fits) + "\n")
    print("  saved sweep_fits.txt")


if __name__ == "__main__":
    main()
