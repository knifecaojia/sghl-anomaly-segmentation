#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Line B weakly-supervised sample-efficiency curve (paper candidate figure).

x = k-shot (1 / 5 / 10 real labeled anomalies per class, log axis),
y = macro P-F1_max over 15 MVTec-AD categories (eval set excludes shot images).

Main lines: B-base (Dice-only) vs B-hic (Dice + HIC, the Line-B winner arm),
mean over available seeds with min/max error bars (k=5/10: 3 seeds for
base/hic, 2 for halo; k=1: seed=1 only). B-halo (Dice+HS+HIC, naive transfer
from the synthetic line) is shown as a light dashed reference. B-gated
(Dice+HIC+g*HS, seed=1 only) is shown as diamond scatter. A dotted gray line
marks the 0-shot start point (Line A halo stage-2 ckpt, P-F1=0.6177 on the
full test set -- slightly different eval population, reference only).

Data: lineB_weak/2026*_lineB_inpformer_mvtecad_k{1,5,10}_{arm}_seed{1,2,3}.json
Style: matches scripts/plot_c3_mechanism.py (daimon_runtime.setup_plot,
Okabe-Ito colorblind-safe palette, 300 dpi, English labels).

Output: figures/fig_lineB_sample_efficiency.png (300 dpi)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]          # .../paper/halo_ad
DATA = ROOT / "lineB_weak"
FIG = ROOT / "figures"
OUT = FIG / "fig_lineB_sample_efficiency.png"

# Okabe-Ito palette, consistent with fig_c3_mechanism
C_BASE = "#0072B2"   # blue       (B-base, Dice-only)
C_HIC = "#009E73"    # green      (B-hic, Dice+HIC -- winner arm)
C_HALO = "#D55E00"   # vermillion (B-halo, Dice+HS+HIC -- reference)
C_GATE = "#CC79A7"   # purple     (B-gated, Dice+HIC+g*HS -- seed=1 only)
C_REF = "#7F7F7F"    # gray       (0-shot start point)

KS = [1, 5, 10]
START_PF1 = 0.6177   # Line A halo stage-2 ckpt, full-test macro P-F1


def load_pf1():
    """-> {(arm, k): [pf1_seed1, pf1_seed2, pf1_seed3(若存在)...]}"""
    out = {}
    for f in glob.glob(str(DATA / "2026*_lineB_inpformer_mvtecad_k*_seed*.json")):
        if "smoke" in f:
            continue
        d = json.load(open(f, encoding="utf-8"))
        key = (d["arm"], d["k_shot"])
        out.setdefault(key, {})[d["seed"]] = d["macro"]["P-F1_max"]
    return {k: [v[s] for s in sorted(v)] for k, v in out.items()}


def series(ax, pf1, arm, color, ls, lw, marker, label, alpha=1.0, zorder=3):
    ks = [k for k in KS if (arm, k) in pf1]
    mean = [np.mean(pf1[(arm, k)]) for k in ks]
    # min/max error bars only where two seeds exist
    lo = [m - min(pf1[(arm, k)]) for k, m in zip(ks, mean)]
    hi = [max(pf1[(arm, k)]) - m for k, m in zip(ks, mean)]
    ax.errorbar(ks, mean, yerr=[lo, hi], color=color, ls=ls, lw=lw,
                marker=marker, ms=6, capsize=3.5, capthick=1.2,
                alpha=alpha, label=label, zorder=zorder)
    return ks, mean


def main():
    setup_plot()
    pf1 = load_pf1()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))

    # 0-shot reference (start ckpt, full test set) -- label at the right end
    ax.axhline(START_PF1, ls=":", lw=1.2, color=C_REF, zorder=1)
    ax.annotate(f"0-shot start ckpt = {START_PF1:.4f}\n(full test, reference)",
                xy=(12.3, START_PF1), xytext=(0, 7), textcoords="offset points",
                ha="right", fontsize=7.5, color=C_REF)

    # main arms first (legend order), halo reference last
    ks_b, m_b = series(ax, pf1, "base", C_BASE, "-", 2.0, "o",
                       "B-base (Dice)")
    ks_h, m_h = series(ax, pf1, "hic_only", C_HIC, "-", 2.0, "o",
                       "B-hic (Dice+HIC, ours)")
    series(ax, pf1, "halo", C_HALO, "--", 1.4, "s", "B-halo (Dice+HS+HIC)",
           alpha=0.45, zorder=2)
    # gated arm: seed=1 only, diamond scatter with slight log-x offset
    gx = [k * 1.09 for k in KS if ("gated", k) in pf1]
    gy = [np.mean(pf1[("gated", k)]) for k in KS if ("gated", k) in pf1]
    ax.plot(gx, gy, "D", color=C_GATE, ms=6.5, ls="none", zorder=4,
            label="B-gated (Dice+HIC+g·HS, seed=1)")
    # explicit legend order (Line2D would otherwise jump ahead of containers)
    handles, labels_ = ax.get_legend_handles_labels()
    order = ["B-base (Dice)", "B-hic (Dice+HIC, ours)",
             "B-gated (Dice+HIC+g·HS, seed=1)", "B-halo (Dice+HS+HIC)"]
    hl = dict(zip(labels_, handles))
    ax.legend([hl[o] for o in order if o in hl],
              [o for o in order if o in hl],
              fontsize=8.5, loc="lower right", framealpha=0.9)

    # value annotations (mean), per-point offsets to avoid label collisions
    offs = {("base", 1): (0, -14), ("base", 5): (0, -16), ("base", 10): (11, 4),
            ("hic_only", 1): (0, 8), ("hic_only", 5): (0, 10), ("hic_only", 10): (0, 10),
            ("halo", 1): (8, -12)}
    ha = {("base", 10): "left"}
    for arm, c in [("base", C_BASE), ("hic_only", C_HIC), ("halo", C_HALO)]:
        for k in KS:
            if (arm, k) not in pf1 or (arm, k) not in offs:
                continue
            m = np.mean(pf1[(arm, k)])
            ax.annotate(f"{m:.4f}", xy=(k, m), xytext=offs[(arm, k)],
                        textcoords="offset points",
                        ha=ha.get((arm, k), "center"), fontsize=7.5,
                        color=c, alpha=0.9 if arm != "halo" else 0.55)
    # gated 数值标签只标 k=10 (k=5 的标签必然压线, 数值由表格承载)
    if gy:
        ax.annotate(f"{gy[-1]:.4f}", xy=(gx[-1], gy[-1]), xytext=(13, -3),
                    textcoords="offset points", ha="left", fontsize=7.5,
                    color=C_GATE, alpha=0.9)
    # seed-coverage note, top-left free area
    ax.text(0.03, 0.955, "error bars = min/max over seeds (base/B-hic: 3; B-halo: 2)\n"
                         "k=1 and B-gated: seed=1 only",
            transform=ax.transAxes, fontsize=7.5, color="#555555", va="top")

    ax.set_xscale("log")
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlim(0.9, 14.0)
    ax.set_xlabel("k-shot  (real labeled anomalies per class)")
    ax.set_ylabel("P-F1$_\\mathrm{max}$  (macro, 15 classes)")
    ax.set_ylim(0.585, 0.705)
    ax.grid(axis="y", which="major", ls=":", lw=0.6, alpha=0.6, zorder=0)

    FIG.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
