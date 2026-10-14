#!/usr/bin/env python3
"""Figure 11 (rw_ci.pdf, with 95% CI error bars) and Table 9 (mean +/- 95% t-CI)
from N repetitions of the eight Category-A BGV speed and memory tags
(s_{rf,gbdt}_bgv_{ud,od}, m_{rf,gbdt}_bgv_{ud,od}).

Outputs (into --outdir):
  rw_ci.pdf               Figure 11: time (stage-stacked) and peak memory, normalized per model to COPSE
  tab_absolute_ci.md      Table 9 cells, mean +/- CI (latency s, peak memory GB) + per-model speedups
  tab_absolute_ci.tex     the same as a LaTeX tabular
  tab_absolute_paper.tex  the body rows exactly as typeset in the paper (CI only where it shows)
  headline_ci.txt         geomean OD speedup and memory footprint vs COPSE and VESTA, +/- CI

    python reproduce/make_end2end.py [--results data/paper_results] [--outdir reproduce/figures_paper]
"""
import argparse, glob, math, os, sys
from pathlib import Path
from statistics import mean, stdev
import numpy as np

HERE = Path(__file__).resolve().parent
def _add_repo(p):
    sys.path.insert(0, str(Path(p) / "reproduce"))
    sys.path.insert(0, str(Path(p) / "src"))

# t_{0.975, df=n-1} for small n
T_CRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
          9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201}

DATASETS = ["income", "soccer", "iris", "mnist"]
PLAN = [("rf", "income", "Income\n(RF)"), ("gbdt", "income", "Income\n(GBDT)"),
        ("rf", "soccer", "Soccer\n(RF)"), ("gbdt", "soccer", "Soccer\n(GBDT)"),
        ("rf", "iris", "Iris\n(RF)"),     ("gbdt", "iris", "Iris\n(GBDT)"),
        ("rf", "mnist", "MNIST\n(RF)"),   ("gbdt", "mnist", "MNIST\n(GBDT)")]


def ci95(vals):
    """-> (mean, half-width, n).  n<2 -> half=0 (single measurement)."""
    vals = [v for v in vals if v is not None and v == v]
    n = len(vals)
    if n == 0: return (float("nan"), 0.0, 0)
    if n == 1: return (vals[0], 0.0, 1)
    m = mean(vals); sd = stdev(vals)
    half = T_CRIT.get(n, 2.776) * sd / math.sqrt(n)
    return (m, half, n)


def _load_tag(hj, tag, ext, loader):
    """Every repetition of a tag in the results directory: <tag>__<label>__rep<N><ext> (no __omp files)."""
    reps = sorted(glob.glob(f"{hj}/{tag}__*__rep*{ext}"))
    reps = [r for r in reps if "__omp" not in os.path.basename(r)]
    if reps:
        return [loader(r) for r in reps], os.path.basename(reps[0]).split("__")[1]
    return [], None


# Re-runs that supersede some models of a tag: {tag: (rerun_tag, [models])}. Empty: every cell comes
# from the tag itself.
RERUNS = {}


def load_reps(hj, tag, ext, loader):
    """All reps for a tag in the results directory (no __ompN files).
    Models listed in RERUNS are taken from the re-run tag when it exists: rep i of the re-run replaces
    that model in rep i of the original (cycling if the counts differ), so per-rep geomeans stay intact."""
    reps, part = _load_tag(hj, tag, ext, loader)
    if tag in RERUNS:
        rtag, models = RERUNS[tag]
        new, _ = _load_tag(hj, rtag, ext, loader)
        if new:
            if len(new) != len(reps):
                print(f"  [rerun] {rtag}: {len(new)} reps vs {len(reps)} in {tag}; cycling the re-run reps")
            for i, r in enumerate(reps):
                for m in models:
                    if m in new[i % len(new)]:
                        r[m] = new[i % len(new)][m]
            print(f"  [rerun] {tag}: {', '.join(models)} taken from {rtag} ({len(new)} reps)")
    return reps, part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(HERE.parent))  # this file lives in reproduce/ -> parent = repo root
    ap.add_argument("--results", dest="data_dir", default=None, help="results directory (default data/paper_results)")
    ap.add_argument("--outdir", default=None, help="output directory (default reproduce/figures_paper)")
    a = ap.parse_args()
    if a.data_dir is None: a.data_dir = str(Path(a.repo) / "data" / "paper_results")
    if a.outdir is None:   a.outdir = str(Path(a.repo) / "reproduce" / "figures_paper")
    _add_repo(a.repo)
    import figstyle as fs
    from make_figures import (load_speed_csv, load_wide, _stage3, PAPER_STAGES,
                              STAGE_COLOR, COMBINED_ORDER, COMBINED_HATCH, _gmean)
    fs.use_style()
    hj = a.data_dir; out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    def stot(entry):  # total seconds of one proto's stage dict
        return sum(_stage3(entry).values())

    # ---- load reps per (kind, tag) ----
    S = {}   # S[(kind,'od'/'ud')] = (list of rep dicts, part)
    W = {}   # W[(kind,'od'/'ud')] = (list of wide rep dicts, part)
    for kind in ("rf", "gbdt"):
        for cfg in ("ud", "od"):
            S[(kind, cfg)] = load_reps(hj, f"s_{kind}_bgv_{cfg}", ".csv", load_speed_csv)
            W[(kind, cfg)] = load_reps(hj, f"m_{kind}_bgv_{cfg}", ".wide.csv", load_wide)

    if not any(reps for reps, _ in S.values()):
        print(f"  [skip] Table 9 / rw_ci.pdf: no s_{{rf,gbdt}}_bgv_{{ud,od}} CSVs in {hj}"); return

    # ==================== Table 9 with CIs ====================
    # per (kind,ds): latency & mem for COPSE, VESTA, 1MM_UD, 1MM_OD (mean +/- CI over reps)
    cells = {}  # cells[(kind,ds)] = dict(lat=..., mem=...) each a dict name-> (mean,ci,n)
    nreps = {}
    for kind, ds, _lab in PLAN:
        sod, sod_p = S[(kind, "od")]; sud, sud_p = S[(kind, "ud")]
        wod, _ = W[(kind, "od")]; wud, _ = W[(kind, "ud")]
        def lat(reps, ds, proto):
            return [stot(r[ds][proto]) for r in reps if ds in r and proto in r[ds]]
        def memg(reps, ds, proto):
            # GNU time reports max RSS in KiB; convert to decimal GB (1e9 B), the unit of Table 8
            return [r[ds][proto] * 1024 / 1e9 for r in reps if ds in r and proto in r[ds]]
        lat_cells = {"COPSE": ci95(lat(sod, ds, "COPSE")), "VESTA": ci95(lat(sod, ds, "VESTA")),
                     "1MM (UD)": ci95(lat(sud, ds, "IMM")), "1MM (OD)": ci95(lat(sod, ds, "IMM"))}
        mem_cells = {"COPSE": ci95(memg(wod, ds, "COPSE")), "VESTA": ci95(memg(wod, ds, "VESTA")),
                     "1MM (UD)": ci95(memg(wud, ds, "1MM")), "1MM (OD)": ci95(memg(wod, ds, "1MM"))}
        cells[(kind, ds)] = dict(lat=lat_cells, mem=mem_cells, part=sod_p)
        nreps[(kind, ds)] = (len(sod), len(sud), len(wod), len(wud))

    # ---- markdown table ----
    md = ["# Table 9 with 95% t-CIs over N repetitions\n",
          f"data: `{hj}`  | reps (s_od,s_ud,m_od,m_ud) per model shown as n=…\n",
          "Latency (s) and peak memory (GB), mean ± 95% CI over reps. Speedup = COPSE/1MM_OD.\n",
          "| Model | n | COPSE lat | VESTA lat | 1MM_UD lat | 1MM_OD lat | COPSE mem | VESTA mem | 1MM_UD mem | 1MM_OD mem | OD speedup |",
          "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    def cfmt(c, prec=1):
        m, h, n = c
        if m != m: return "—"
        return f"{m:.{prec}f}±{h:.{prec}f}" if h > 0 else f"{m:.{prec}f}"
    for kind, ds, lab in PLAN:
        c = cells[(kind, ds)]; L = c["lat"]; M = c["mem"]
        nn = [x for x in nreps[(kind, ds)] if x]          # reps of the tags present (some tags may be missing)
        n = f"{min(nn)}" if nn else "0"
        if nn and len(nn) < 4: n += f" ({len(nn)}/4 tags)"
        cs = L["COPSE"][0]; io = L["1MM (OD)"][0]
        spd = cs / io if io else float("nan")
        md.append(f"| {kind}/{ds} | {n} | {cfmt(L['COPSE'],0)} | {cfmt(L['VESTA'],0)} | "
                  f"{cfmt(L['1MM (UD)'],0)} | {cfmt(L['1MM (OD)'],1)} | {cfmt(M['COPSE'],1)} | "
                  f"{cfmt(M['VESTA'],1)} | {cfmt(M['1MM (UD)'],1)} | {cfmt(M['1MM (OD)'],1)} | {spd:.2f}× |")
    (out / "tab_absolute_ci.md").write_text("\n".join(md) + "\n")

    # ---- LaTeX (mean {\footnotesize $\pm$ci}) ----
    def tex(c, prec=1):
        m, h, n = c
        if m != m: return "--"
        s = f"{m:.{prec}f}"
        return s + (rf"\,{{\footnotesize$\pm${h:.{prec}f}}}" if h > 0 else "")
    tl = [r"% Table 9 with 95% CIs -- generated by reproduce/make_end2end.py",
          r"\begin{tabular}{ll rrrr rrrr}", r"\toprule",
          r" & & \multicolumn{4}{c}{Latency (s)} & \multicolumn{4}{c}{Peak memory (GB)} \\",
          r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
          r"Dataset & Model & COPSE & VESTA & 1MM$_{\rm UD}$ & 1MM$_{\rm OD}$ & COPSE & VESTA & 1MM$_{\rm UD}$ & 1MM$_{\rm OD}$ \\",
          r"\midrule"]
    dsname = {"income": "Income", "soccer": "Soccer", "iris": "Iris", "mnist": "MNIST"}
    for kind, ds, lab in PLAN:
        c = cells[(kind, ds)]; L = c["lat"]; M = c["mem"]
        tl.append(f"{dsname[ds]} & {kind.upper()} & {tex(L['COPSE'],0)} & {tex(L['VESTA'],0)} & "
                  f"{tex(L['1MM (UD)'],0)} & {tex(L['1MM (OD)'],1)} & {tex(M['COPSE'],1)} & "
                  f"{tex(M['VESTA'],1)} & {tex(M['1MM (UD)'],1)} & {tex(M['1MM (OD)'],1)} \\\\")
    tl += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_absolute_ci.tex").write_text("\n".join(tl) + "\n")

    # ---- the paper's Table 9 rows (mean, with ${\pm}$CI only where it is nonzero at the shown precision) ----
    def ptex(c, prec):
        m, h, n = c
        if m != m: return "--"
        return f"${m:.{prec}f}{{\\pm}}{h:.{prec}f}$" if round(h, prec) > 0 else f"${m:.{prec}f}$"
    pl = [r"% Table 9 body rows -- auto-generated by make_end2end.py (COPSE, VESTA at OD)"]
    for kind, ds, lab in PLAN:
        c = cells[(kind, ds)]; L = c["lat"]; M = c["mem"]
        name = f"{dsname[ds]} ({kind.upper()})"
        lat_prec = {"COPSE": 0, "VESTA": 0, "1MM (UD)": 0, "1MM (OD)": 1}
        row = [ptex(L[k], lat_prec[k]) for k in ("COPSE", "VESTA", "1MM (UD)", "1MM (OD)")]
        row += [ptex(M[k], 1) for k in ("COPSE", "VESTA", "1MM (UD)", "1MM (OD)")]
        pl.append(f"{name:<14}& " + " & ".join(row) + r" \\")
    nmin = min(min(v) for v in nreps.values()); nmax = max(max(v) for v in nreps.values())
    pl.append(f"% reps per cell: {nmin}..{nmax}; max memory CI half-width: "
              f"{max(cells[(k, d)]['mem'][p][1] for k, d, _ in PLAN for p in cells[(k, d)]['mem']):.3f} GB")
    (out / "tab_absolute_paper.tex").write_text("\n".join(pl) + "\n")

    # ==================== HEADLINE geomean OD speedup +/- CI ====================
    # per rep: geomean over 8 models of (COPSE_od/1MM_od).  Need paired reps -> use min rep count.
    def per_rep_geo(metric):
        # metric(kind,ds,ri) -> value or None ; returns list over reps of geomean across 8 models
        nmin = min(len(S[(k, 'od')][0]) for k in ('rf', 'gbdt'))
        res = []
        for ri in range(nmin):
            vals = []
            for kind, ds, _l in PLAN:
                v = metric(kind, ds, ri)
                if v and v > 0: vals.append(v)
            if len(vals) == len(PLAN):
                res.append(_gmean(vals))
        return res
    def spd_metric(kind, ds, ri):
        sod = S[(kind, 'od')][0]
        if ri >= len(sod) or ds not in sod[ri]: return None
        c = stot(sod[ri][ds]["COPSE"]); i = stot(sod[ri][ds]["IMM"])
        return c / i if i else None
    def mem_metric(kind, ds, ri):
        wod = W[(kind, 'od')][0]
        if ri >= len(wod) or ds not in wod[ri]: return None
        c = wod[ri][ds].get("COPSE"); i = wod[ri][ds].get("1MM")
        return (i / c) if (c and i) else None
    geo_spd = per_rep_geo(spd_metric); geo_mem = per_rep_geo(mem_metric)
    gs = ci95(geo_spd); gm = ci95(geo_mem)
    hl = [f"Headline (geomean over 8 Category-A models, OD, BGV), N-rep 95% CI:"]
    if gs[2]:
        hl.append(f"  OD speedup vs COPSE : {gs[0]:.2f} ± {gs[1]:.2f} ×   (n={gs[2]} reps)   [paper 6.7×]")
    else:
        hl.append("  OD speedup vs COPSE : needs both s_rf_bgv_od and s_gbdt_bgv_od (all 8 models)   [paper 6.7×]")
    if gm[2]:
        hl.append(f"  mem footprint (1MM_OD/COPSE) : {gm[0]*100:.1f} ± {gm[1]*100:.1f} %   -> reduction {100-gm[0]*100:.1f} ± {gm[1]*100:.1f} %   [paper 13% / 87%]")
    else:
        hl.append("  mem footprint (1MM_OD/COPSE) : needs both m_rf_bgv_od and m_gbdt_bgv_od (all 8 models)   [paper 13% / 87%]")
    (out / "headline_ci.txt").write_text("\n".join(hl) + "\n")
    print("\n".join(hl))

    # ==================== Figure 11 (rw_ci.pdf) with error bars ====================
    # Build normalized per-rep values, then mean bar heights + CI of the total per bar.
    # names: COPSE(ud)/COPSE_ud, VESTA(ud)/COPSE_ud, 1MM(UD)=IMM_ud/COPSE_ud, 1MM(OD)=IMM_od/COPSE_ud
    src_time = {"COPSE": ("ud", "COPSE"), "VESTA": ("ud", "VESTA"),
                "1MM (UD)": ("ud", "IMM"), "1MM (OD)": ("od", "IMM")}
    src_mem = {"COPSE": ("ud", "COPSE"), "VESTA": ("ud", "VESTA"),
               "1MM (UD)": ("ud", "1MM"), "1MM (OD)": ("od", "1MM")}

    # per model: mean per-stage normalized (for stack) + list of per-rep normalized totals (for CI)
    def model_time(kind, ds):
        sud = S[(kind, 'ud')][0]; sod = S[(kind, 'od')][0]
        R = min(len(sud), len(sod))
        stage_means = {nm: {s: [] for s in PAPER_STAGES} for nm in COMBINED_ORDER}
        totals = {nm: [] for nm in COMBINED_ORDER}
        for ri in range(R):
            ru, ro = sud[ri], sod[ri]
            if ds not in ru or ds not in ro: continue
            denom = stot(ru[ds]["COPSE"]) or 1e-9
            for nm in COMBINED_ORDER:
                cfg, proto = src_time[nm]; src = ru if cfg == "ud" else ro
                if ds not in src or proto not in src[ds]: continue
                sv = _stage3(src[ds][proto])
                for s in PAPER_STAGES:
                    stage_means[nm][s].append(sv.get(s, 0.0) / denom)
                totals[nm].append(sum(sv.values()) / denom)
        sm = {nm: {s: (mean(stage_means[nm][s]) if stage_means[nm][s] else 0.0) for s in PAPER_STAGES}
              for nm in COMBINED_ORDER}
        tci = {nm: ci95(totals[nm]) for nm in COMBINED_ORDER}
        return sm, tci
    def model_mem(kind, ds):
        wud = W[(kind, 'ud')][0]; wod = W[(kind, 'od')][0]
        R = min(len(wud), len(wod))
        vals = {nm: [] for nm in COMBINED_ORDER}
        for ri in range(R):
            ru, ro = wud[ri], wod[ri]
            if ds not in ru or ds not in ro: continue
            denom = ru[ds].get("COPSE") or 1e-9
            for nm in COMBINED_ORDER:
                cfg, proto = src_mem[nm]; src = ru if cfg == "ud" else ro
                if ds in src and proto in src[ds]:
                    vals[nm].append(src[ds][proto] / denom)
        return {nm: ci95(vals[nm]) for nm in COMBINED_ORDER}

    time_stack, time_tci, mem_ci, labs = [], [], [], []
    for kind, ds, lab in PLAN:
        sm, tci = model_time(kind, ds)
        if not any(tci[nm][2] for nm in COMBINED_ORDER):  # no reps
            continue
        mc = model_mem(kind, ds)
        time_stack.append(sm); time_tci.append(tci); mem_ci.append(mc); labs.append(lab)
    n = len(labs)
    if n == 0:
        print("  [skip] rw_ci.pdf: needs the od AND ud speed tag of at least one family (s_rf_bgv_od + s_rf_bgv_ud, ...)"); return

    # geomean cluster (per-rep geomean already computed as gs/gm for the OD speedup; for the bar we
    # need geomean of the normalized total per name)
    def geo_name_time(nm):
        # per-rep geomean over models of normalized total for this name, then mean+CI
        nmin = min(len(S[(k, 'ud')][0]) for k in ('rf', 'gbdt'))
        per = []
        for ri in range(nmin):
            vs = []
            for kind, ds, _l in PLAN:
                ru = S[(kind, 'ud')][0][ri] if ri < len(S[(kind, 'ud')][0]) else None
                ro = S[(kind, 'od')][0][ri] if ri < len(S[(kind, 'od')][0]) else None
                if not ru or not ro or ds not in ru or ds not in ro: continue
                denom = stot(ru[ds]["COPSE"]) or 1e-9
                cfg, proto = src_time[nm]; src = ru if cfg == "ud" else ro
                if ds in src and proto in src[ds]:
                    vs.append(sum(_stage3(src[ds][proto]).values()) / denom)
            if len(vs) == len(PLAN): per.append(_gmean(vs))
        return ci95(per)
    def geo_name_mem(nm):
        nmin = min(len(W[(k, 'ud')][0]) for k in ('rf', 'gbdt'))
        per = []
        for ri in range(nmin):
            vs = []
            for kind, ds, _l in PLAN:
                ru = W[(kind, 'ud')][0][ri] if ri < len(W[(kind, 'ud')][0]) else None
                ro = W[(kind, 'od')][0][ri] if ri < len(W[(kind, 'od')][0]) else None
                if not ru or not ro or ds not in ru or ds not in ro: continue
                denom = ru[ds].get("COPSE") or 1e-9
                cfg, proto = src_mem[nm]; src = ru if cfg == "ud" else ro
                if ds in src and proto in src[ds]:
                    vs.append(src[ds][proto] / denom)
            if len(vs) == len(PLAN): per.append(_gmean(vs))
        return ci95(per)

    geo_time = {nm: geo_name_time(nm) for nm in COMBINED_ORDER}
    geo_mem = {nm: geo_name_mem(nm) for nm in COMBINED_ORDER}
    have_mem = all(mem_ci[i][nm][2] for i in range(n) for nm in COMBINED_ORDER) and \
               all(geo_mem[nm][2] for nm in COMBINED_ORDER)

    if have_mem:
        fig, (axT, axM) = fs.new_fig(w=fs.COL_W, h=1.85, ncols=2)
        axes = [(axT, "time"), (axM, "mem")]
    else:
        fig, axT = fs.new_fig(w=fs.COL_W, h=1.85)
        axes = [(axT, "time")]
    bar_w, group = 1.0, 4.7
    idx = np.arange(n + 1) * group
    idx[n] += 1.3  # extra gap so the Geomean cluster clears MN GBDT (both panels use this idx)
    _ds = {"Income": "Inc", "Soccer": "Soc", "MNIST": "MN", "Iris": "Iris"}
    def _abbr(lab):
        ds, md_ = (lab.split("\n") + [""])[:2]
        return f"{_ds.get(ds, ds)}\n{md_.strip('()')}"
    xlabels = [_abbr(l) for l in labs] + ["Geo\nMean"]

    for ax, kind in axes:
        for si, nm in enumerate(COMBINED_ORDER):
            xs = idx + si * bar_w
            if kind == "time":
                bottoms = np.zeros(n + 1)
                for stage in PAPER_STAGES:
                    vals = np.array([time_stack[i][nm][stage] for i in range(n)] + [geo_time[nm][0] *
                             (mean([time_stack[i][nm][stage] / (sum(time_stack[i][nm].values()) or 1e-9)
                                    for i in range(n)]))])
                    ax.bar(xs, vals, bar_w, bottom=bottoms, color=STAGE_COLOR[stage],
                           edgecolor=fs.INK, linewidth=0.5, hatch=COMBINED_HATCH[nm] or None)
                    bottoms += vals
                tot = [sum(time_stack[i][nm].values()) for i in range(n)] + [geo_time[nm][0]]
                err = [time_tci[i][nm][1] for i in range(n)] + [geo_time[nm][1]]
            else:
                tot = [mem_ci[i][nm][0] for i in range(n)] + [geo_mem[nm][0]]
                err = [mem_ci[i][nm][1] for i in range(n)] + [geo_mem[nm][1]]
                ax.bar(xs, tot, bar_w, color="white", edgecolor=fs.INK, linewidth=0.5,
                       hatch=COMBINED_HATCH[nm] or None)
            # error bars on the totals (only where CI>0)
            ax.errorbar(xs, tot, yerr=err, fmt="none", ecolor=fs.INK, elinewidth=0.6,
                        capsize=1.5, capthick=0.6)
        # geomean 1MM annotations (speedup x / footprint %)
        gv = geo_time if kind == "time" else geo_mem
        is_time = kind == "time"
        for nm, ytxt in (("1MM (UD)", 0.62), ("1MM (OD)", 0.34)):
            v = gv[nm][0]
            if not v or v <= 0: continue
            txt = rf"{1/v:.1f}$\times$" if is_time else f"{v*100:.0f}%"
            bx = idx[n] + COMBINED_ORDER.index(nm) * bar_w
            ax.annotate(txt, xy=(bx + 0.35, v + 0.02), xytext=(bx + 0.35, ytxt), ha="center",
                        va="bottom", rotation=90, fontsize=6, color=fs.INK,
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=fs.INK, shrinkA=1, shrinkB=1))
        ax.set_ylabel("Norm. time" if is_time else "Norm. memory", fontsize=fs.BODY_PT - 1)
        ax.set_xticks(idx + 1.5 * bar_w)
        ax.set_xticklabels(xlabels, fontsize=fs.BODY_PT - 3)
        ax.tick_params(axis="y", labelsize=fs.BODY_PT - 3)
        sep_x = (idx[n - 1] + (len(COMBINED_ORDER) - 1) * bar_w + bar_w / 2 + idx[n] - bar_w / 2) / 2
        ax.margins(x=0.02)
        ax.axvline(sep_x, color=fs.INK, lw=0.6, ls=(0, (2, 2)))

    # legend (reuse make_figures' combined legend via a light re-impl)
    from matplotlib.patches import Patch
    stage_h = [Patch(facecolor=STAGE_COLOR[s], edgecolor=fs.INK, linewidth=0.5,
                     label={"Path Construct": "Path Constr."}.get(s, s)) for s in PAPER_STAGES]
    set_h = [Patch(facecolor="white", edgecolor=fs.INK, linewidth=0.5,
                   hatch=(COMBINED_HATCH[k] or None), label=k) for k in COMBINED_ORDER]
    fig.legend(handles=stage_h + set_h, loc="outside upper center", ncol=7,
               fontsize=fs.BODY_PT - 3, columnspacing=1.0, handlelength=1.2,
               handletextpad=0.4, borderaxespad=0.2)
    from matplotlib.text import Text
    fig.canvas.draw()
    for t in fig.findobj(Text):
        t.set_fontweight("bold")
    fs.save(fig, out / "rw_ci.pdf")
    print(f"  saved rw_ci.pdf  (Figure 11 with 95% CI error bars, {n} models, "
          f"reps: time={time_tci[0]['COPSE'][2] if n else 0}, mem={'y' if have_mem else 'n'})")


if __name__ == "__main__":
    main()
