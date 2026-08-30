#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C3 机理主图出版级重绘 (REVIEW_V1 P2 #11).

三联: (a) 能量-距离衰减 E(d) (双臂×双支路, log-y, 宏平均);
      (b) 区域能量 (interior/halo/far-bg/normal, log-y, 塌缩 vs 救活);
      (c) 代表样例 (screw_005 小缺陷 + transistor_007 大缺陷):
          输入 | GT | baseline 能量 | halo 能量 (1-99 百分位拉伸).

数据: results/lambda_shrinkage/lambda_curves.json (Line A INP-Former 载体),
      results/overfire_check/overfire_stats.json,
      results/energy_samples/{baseline,halo}/*.npy,
      figures/assets/*_input.png.
输出: figures/fig_c3_mechanism.pdf (矢量) + .png (600 dpi).
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

C_BASE = '#7f7f7f'
C_HALO = '#c0392b'
C_COMB_BASE = '#b8b8b8'
C_COMB_HALO = '#e67e7a'

# ---- data -------------------------------------------------------------
lc = json.load(open(os.path.join(RES, 'lambda_shrinkage', 'lambda_curves.json')))
ov = json.load(open(os.path.join(RES, 'overfire_check', 'overfire_stats.json')))


def macro_curve(branch, arm):
    d = lc[branch][arm]['per_category_E_of_d']
    return np.mean(np.array(list(d.values()), dtype=np.float64), axis=0)


dist = np.array(lc['seghead']['distances'], dtype=np.float64)

# ---- figure -----------------------------------------------------------
fig = plt.figure(figsize=(16 / 2.54, 11.6 / 2.54))  # 16cm x 11.6cm
gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.82], hspace=0.42, wspace=0.3)

# (a) energy-distance decay ------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
for branch, ls in (('seghead', '-'), ('combined', '--')):
    for arm, col in (('baseline', C_BASE if branch == 'seghead' else C_COMB_BASE),
                     ('halo', C_HALO if branch == 'seghead' else C_COMB_HALO)):
        y = macro_curve(branch, arm)
        lw = 1.5 if branch == 'seghead' else 0.9
        label = f'{arm} ({branch})'
        ax.semilogy(dist, np.clip(y, 1e-24, None), ls, color=col, lw=lw, label=label)
ax.set_xlabel('Distance to GT boundary $d$ (px)')
ax.set_ylabel('Mean energy $E(d)$')
ax.set_xlim(1, 50)
ax.set_ylim(1e-21, 1e0)
ax.annotate('collapsed\n($\\sim\\!10^{-20}$)', xy=(26, 3e-20), fontsize=7.5,
            color=C_BASE, ha='center', va='bottom')
ax.annotate('$E(50)/E(1){=}0.37$', xy=(30, 1.5e-1), fontsize=7.5, color=C_HALO)
ax.legend(frameon=False, loc='lower left', ncol=2, handlelength=1.6,
          columnspacing=0.9)
ax.text(0.02, 1.04, '(a)', transform=ax.transAxes, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (b) region energies ------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
arms = ov['macro_summary']['arms']
regions = [('seghead_anom_interior_mean', 'interior'),
           ('seghead_anom_halo_mean', 'halo band'),
           ('seghead_anom_farbg_mean', 'far bg'),
           ('seghead_normal_mean', 'normal img')]
x = np.arange(len(regions))
w = 0.38
base_vals = [max(arms['baseline'][k], 1e-24) for k, _ in regions]
halo_vals = [arms['halo'][k] for k, _ in regions]
ax.bar(x - w / 2, base_vals, w, color=C_BASE, label='baseline (collapsed)')
ax.bar(x + w / 2, halo_vals, w, color=C_HALO, label='halo (HIC)')
ax.set_yscale('log')
ax.set_ylim(1e-21, 1e0)
ax.set_xticks(x)
ax.set_xticklabels([r for _, r in regions])
ax.set_ylabel('Seg-head mean energy')
for xi, v in zip(x - w / 2, base_vals):
    ax.text(xi, v * 3, r'$\sim\!10^{-20}$', ha='center', fontsize=6.5, color='#444')
for xi, v in zip(x + w / 2, halo_vals):
    ax.text(xi, v * 1.6, f'{v:.2f}', ha='center', fontsize=6.5, color=C_HALO)
ax.legend(frameon=False, loc='upper right')
ax.text(0.02, 1.04, '(b)', transform=ax.transAxes, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (c) representative maps --------------------------------------------------
samples = ['screw_005', 'transistor_007']
inner = gridspec.GridSpecFromSubplotSpec(len(samples), 4, subplot_spec=gs[1, :],
                                         wspace=0.06, hspace=0.12)
for r, s in enumerate(samples):
    inp = plt.imread(os.path.join(HERE, 'assets', f'{s.split("_")[0]}_{s.split("_")[1]}_input.png'))
    gt = np.load(os.path.join(RES, 'energy_samples', 'baseline', f'{s}_gt.npy'))
    eb = np.load(os.path.join(RES, 'energy_samples', 'baseline', f'{s}_energy.npy')).astype(np.float32)
    eh = np.load(os.path.join(RES, 'energy_samples', 'halo', f'{s}_energy.npy')).astype(np.float32)

    def stretch(e):
        lo, hi = np.percentile(e, 1), np.percentile(e, 99)
        return np.clip((e - lo) / max(hi - lo, 1e-8), 0, 1)

    cols = [(inp, None), (gt, None), (stretch(eb), 'magma'), (stretch(eh), 'magma')]
    for c, (img, cmap) in enumerate(cols):
        ax = fig.add_subplot(inner[r, c])
        ax.imshow(img, cmap=cmap) if img.ndim == 2 else ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        if r == 0:
            titles = ['input', 'GT', 'baseline $E$', 'halo $E$ (HIC)']
            ax.set_title(titles[c], fontsize=8, pad=2.5)
        if c == 0:
            ax.set_ylabel(s.replace('_', ' '), fontsize=7.5, rotation=0,
                          ha='right', va='center', labelpad=16)
fig.text(0.012, 0.015, '(c) energy maps stretched to [1,99] percentiles per image; '
         'baseline is diffuse, halo concentrates on the defect.', fontsize=6.8,
         color='#444')

fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.09)
out_pdf = os.path.join(HERE, 'fig_c3_mechanism.pdf')
out_png = os.path.join(HERE, 'fig_c3_mechanism600.png')
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=600)
print('saved:', out_pdf)
print('saved:', out_png)
