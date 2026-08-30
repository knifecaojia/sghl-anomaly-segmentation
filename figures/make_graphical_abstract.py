#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CVM 图形摘要 (Graphical Abstract) — 投稿物料.

CVM 规范: JPG, 300 dpi, 高:宽 = 8:13。
三联构图 (问题 → 方法 → 结果):
  A COLLAPSE   : Dice/BCE 头塌缩为常数图 (输入/GT/塌缩能量图), 8/9 种子位
  B HIC 方法    : 边界几何示意 (interior/boundary/halo 嵌套 + 1D 能量剖面 + margin)
  C STABILITY  : HIC 能量图聚焦缺陷 + 塌缩计数条形 (8/9 vs 0/12) + 关键数字

数据: results/energy_samples (screw_005), 冻结评估 JSON 汇总。
输出: figures/graphical_abstract.jpg (3900x2400 @300dpi) + 预览 png。
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,
})

W_IN, H_IN = 13, 8          # 13:8 = 长:高, 规范要求 高:长 = 8:13
DPI = 300

C_BASE = '#7f7f7f'
C_HALO = '#c0392b'
C_INT = '#e74c3c'
C_BOUND = '#f39c12'
C_HALOBAND = '#f7dc6f'
C_BG = '#eaf2f8'

fig = plt.figure(figsize=(W_IN, H_IN))
gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.18, 1.02],
                      left=0.015, right=0.985, top=0.86, bottom=0.045,
                      wspace=0.16)

# ---------------- panel titles ----------------
titles = [('A  COLLAPSE', C_BASE), ('B  HIC: supervise boundary geometry', C_HALO),
          ('C  STABILITY', '#1a6180')]
for ax_i, (t, c) in zip(range(3), titles):
    ax = fig.add_subplot(gs[0, ax_i])
    ax.text(0.5, 1.055, t, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=17, fontweight='bold', color=c)

# ================ Panel A: collapse =================
axA = fig.add_subplot(gs[0, 0])
axA.set_xlim(0, 3); axA.set_ylim(0, 3.35); axA.axis('off')

inp = plt.imread(os.path.join(HERE, 'assets', 'screw_005_input.png'))
gt = np.load(os.path.join(RES, 'energy_samples', 'baseline', 'screw_005_gt.npy'))
eb = np.load(os.path.join(RES, 'energy_samples', 'baseline',
                          'screw_005_energy.npy')).astype(np.float32)


def stretch(e):
    lo, hi = np.percentile(e, 1), np.percentile(e, 99)
    return np.clip((e - lo) / max(hi - lo, 1e-8), 0, 1)


cells = [(inp, 'input', None), (gt, 'ground truth', 'gray'),
         (stretch(eb), 'Dice+BCE energy', 'viridis')]
for i, (img, cap, cmap) in enumerate(cells):
    ax = axA.inset_axes([0.02 + i * 0.335, 0.42, 0.30, 0.42], transform=axA.transAxes)
    ax.imshow(img, cmap=cmap) if img.ndim == 2 else ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.1); s.set_color('#333')
    ax.set_title(cap, fontsize=13, pad=4)

axA.text(0.5, 0.34, 'cross-dataset + 0.1–0.3% positive pixels\n'
         r'$\Rightarrow$ head collapses to a constant map',
         ha='center', va='top', fontsize=12.5, color='#333',
         transform=axA.transAxes)
axA.text(0.5, 0.20, r'$E \approx 10^{-20}$ inside defects', ha='center',
         fontsize=13, color=C_BASE, transform=axA.transAxes)
axA.text(0.5, 0.115, r'reweighting / focal: no help', ha='center',
         fontsize=12, color='#666', style='italic', transform=axA.transAxes)
axA.text(0.5, 0.02, r'$\mathbf{8/9}$ dataset$\times$seed slots collapse',
         ha='center', fontsize=15, color=C_BASE, transform=axA.transAxes)

# ================ Panel B: HIC geometry =================
axB = fig.add_subplot(gs[0, 1])
axB.set_xlim(-1.35, 1.35); axB.set_ylim(-0.72, 1.06); axB.axis('off')

cx, cy, r_out = -0.72, 0.42, 0.52
# nested regions
for (r, col, a, ec) in [(r_out, C_BG, 1.0, 'none'),
                        (0.40, C_HALOBAND, 1.0, 'none'),
                        (0.315, C_BOUND, 1.0, 'none'),
                        (0.21, C_INT, 1.0, 'none')]:
    axB.add_patch(Circle((cx, cy), r, facecolor=col, edgecolor=ec, lw=1))
axB.add_patch(Circle((cx, cy), 0.315, fill=False, ec='#555', lw=1.2, ls='--'))
axB.annotate('halo ring\n(dilate$_5$\\textbackslash G)', xy=(cx + 0.30, cy + 0.34),
             xytext=(cx + 0.62, cy + 0.55), fontsize=12.5, color='#8a6d1a',
             arrowprops=dict(arrowstyle='-', color='#8a6d1a', lw=1))
axB.annotate('boundary band', xy=(cx + 0.27, cy - 0.16),
             xytext=(cx + 0.58, cy - 0.34), fontsize=12.5, color='#a06000',
             arrowprops=dict(arrowstyle='-', color='#a06000', lw=1))
axB.annotate('interior\n(erode$_3$)', xy=(cx - 0.08, cy + 0.05),
             xytext=(cx - 1.28, cy + 0.55), fontsize=12.5, color=C_INT,
             arrowprops=dict(arrowstyle='-', color=C_INT, lw=1))
axB.annotate('background', xy=(cx - 0.48, cy - 0.28),
             xytext=(cx - 1.26, cy - 0.62), fontsize=12.5, color='#557',
             arrowprops=dict(arrowstyle='-', color='#557', lw=1))

# 1D energy profile
x = np.linspace(0, 1, 400)
d = np.abs(x - 0.5)
prof = np.exp(-(d / 0.10) ** 2)
prof = 0.06 + 0.94 * prof
axP = axB.inset_axes([0.30, 0.10, 0.62, 0.32], transform=axB.transAxes)
axP.plot(x, prof, color=C_HALO, lw=2.4)
axP.axhline(0.06, color='#99a', lw=0.8, ls=':')
for xx, lab, col in [(0.5, r'$p_\mathcal{I}$', C_INT),
                     (0.575, r'$p_\mathcal{B}$', C_BOUND),
                     (0.685, r'$p_\mathcal{H}$', '#8a6d1a')]:
    yy = 0.06 + 0.94 * np.exp(-((abs(xx - 0.5)) / 0.10) ** 2)
    axP.plot([xx], [yy], 'o', ms=5, color=col)
    axP.annotate(lab, (xx, yy), xytext=(xx + 0.015, yy + 0.14),
                 fontsize=13, color=col)
axP.annotate('', xy=(0.5, 0.52), xytext=(0.575, 0.52),
             arrowprops=dict(arrowstyle='<->', color='#333', lw=1.2))
axP.text(0.5375, 0.58, r'$m$', ha='center', fontsize=13)
axP.set_xticks([]); axP.set_yticks([])
axP.set_ylim(0, 1.35); axP.set_xlim(0.35, 0.95)
for s in ('top', 'right'):
    axP.spines[s].set_visible(False)
axP.set_xlabel('distance to boundary', fontsize=11.5, labelpad=1)
axB.text(0.5, 0.53, r'$\mathcal{L}_{\mathrm{HIC}}=' r'[p_\mathcal{B}-p_\mathcal{I}+m]_+'
         r'-\log\frac{p_\mathcal{I}}{p_\mathcal{I}+p_\mathcal{H}}$',
         ha='center', fontsize=14.5, transform=axB.transAxes,
         bbox=dict(boxstyle='round,pad=0.35', fc='#fdf6ec', ec='#d5c09a'))
axB.text(0.5, 0.015, 'constant map violates every term; frozen DINOv2 + 2.75M head',
         ha='center', fontsize=11.5, color='#555', transform=axB.transAxes,
         style='italic')

# ================ Panel C: stability =================
axC = fig.add_subplot(gs[0, 2])
axC.axis('off')

eh = np.load(os.path.join(RES, 'energy_samples', 'halo',
                          'screw_005_energy.npy')).astype(np.float32)
ax1 = axC.inset_axes([0.30, 0.50, 0.42, 0.44], transform=axC.transAxes)
ax1.imshow(stretch(eh), cmap='viridis')
ax1.set_xticks([]); ax1.set_yticks([])
for s in ax1.spines.values():
    s.set_linewidth(1.1); s.set_color(C_HALO)
ax1.set_title('HIC energy: focused on defect', fontsize=13, pad=4, color=C_HALO)

ax2 = axC.inset_axes([0.10, 0.16, 0.80, 0.24], transform=axC.transAxes)
bars = ax2.bar([0, 1], [8, 0], color=[C_BASE, C_HALO], width=0.5)
ax2.bar([0, 1], [1, 12], bottom=[8, 0],
        color=['#d9d9d9', '#f5c6c0'], width=0.5)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['baseline\narms', 'HIC\nfamily'])
ax2.text(0, 8.45, '8 collapsed', ha='center', fontsize=12.5, color='white',
         fontweight='bold')
ax2.text(1, 12.35, '12 runs, 0 collapse', ha='center', fontsize=12.5,
         color=C_HALO, fontweight='bold')
ax2.set_ylim(0, 14.5); ax2.set_xlim(-0.55, 1.55)
ax2.set_yticks([0, 3, 6, 9, 12])
ax2.tick_params(labelsize=11)
ax2.set_ylabel('dataset×seed slots', fontsize=11.5)
for s in ('top', 'right'):
    ax2.spines[s].set_visible(False)

axC.text(0.5, 0.095, r'$\mathbf{+0.112 / +0.068}$ P-F1 vs. MultiADS (ICCV\'25)'
         '\nzero collapse: 12/12 slots (MVTec · VisA · Real-IAD)',
         ha='center', fontsize=13, color='#1a6180', transform=axC.transAxes)
axC.text(0.5, 0.015, 'tiny-defect domain: adaptive morphology (derived thresholds)',
         ha='center', fontsize=11.5, color='#8a6d1a', transform=axC.transAxes,
         style='italic')

# arrows between panels
for xa in (0.335, 0.665):
    ax = fig.add_axes([xa, 0.44, 0.03, 0.02])
    ax.axis('off')
    ax.add_patch(FancyArrowPatch((0, 0.5), (1, 0.5), arrowstyle='-|>',
                                 mutation_scale=28, color='#555', lw=2.2))

out_jpg = os.path.join(HERE, 'graphical_abstract.jpg')
out_png = os.path.join(HERE, '_ga_preview.png')
fig.savefig(out_jpg, dpi=DPI, facecolor='white',
            pil_kwargs={'quality': 92})
fig.savefig(out_png, dpi=72, facecolor='white')
from PIL import Image
im = Image.open(out_jpg)
print('saved:', out_jpg, im.size, f'{im.size[0]/im.size[1]:.3f} (target 13/8={13/8:.3f})')
