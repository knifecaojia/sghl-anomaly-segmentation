#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G2 门控统计量先验测量 v2 (纯测量, 不训练) -- 修正统计量 s = E_halo - E_bg.

v1 结论: r = E_halo/E_int 不可分 (pooled BC=0.924, 最优 balAcc=0.559) --
real 的 E_halo 与 E_int 同向升高, 比值把信号约掉。判别信息在 halo 带
*绝对* 能量。v2 改测:

    s(img) = E_halo(img) - E_bg(class)

  - E_halo(img): seg 头 sigmoid 输出在该图 halo 带 (dilate(GT,5)-GT) 的均值
  - E_bg(class): 该类全部正常测试图的 seg 输出全局均值 (在线估计, 零标注)

s 不需要 interior -> 规避了 CutPaste 小补丁/小缺陷 erode(3) 后 interior 为空
的退化 (v1 中 synth 每类 1~7/20 退化)。

门控约定 (SGHL v2): g = sigmoid(-(s - tau)/T); s 低 -> g 高 -> HS 全开
(合成异常 halo 带是纯背景, 能量应接近背景 -> s 低); s 高 -> HS 退火
(真实缺陷 halo 带含真实过渡信号 -> s 高)。

测量: Line A halo ckpt + baseline (塌陷) ckpt 对照; real = 15 类全部 1258 张
测试异常图; synth = 训练管线同款 CutPaste 每类 20 张 (RandomState(1))。
同轮也算 r (与 v1 口径一致, 便于同 JSON 对比)。

产出:
  results/gating_prior/gating_prior_s_stats.json  (E_bg + 逐类 s/r 分布 + 可分性)
  results/gating_prior/fig_gating_prior_s.png     (逐类 real vs synth s violin)
  (v1 的 gating_prior_r_stats.json / fig_gating_prior_r.png 保持不动)

运行: cd /data/repo/halo_ad && python gating_prior.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if not hasattr(np, "trapz"):  # adeval compat shim (numpy>=2)
    np.trapz = np.trapezoid

ROOT = Path("/data/repo/halo_ad")
INP = ROOT / "inp-former-pp"
os.chdir(ROOT)
sys.path.insert(0, str(INP))
sys.path.insert(0, str(ROOT / "analysis"))   # run_pass.py (build_model/forward_branches)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import binary_dilation, binary_erosion
from torch.utils.data import DataLoader

import run_pass                      # 复用 build_model / forward_branches / CKPTS
from dataset import get_dataset, get_data_transforms
from utils import setup_seed

DATA = "/data/repo/AD-DINOv3/Data/Industrial_Datasets/MVTechAD"
CATS = run_pass.CATS
OUT_DIR = ROOT / "results" / "gating_prior"
HALO_DIL = 5
INT_ERODE = 3
N_SYNTH = 20
SEED = 1
EPS = 1e-12


# ------------------------------------------------------------------- measure

def region_stats(seg, gt_bool):
    """seg: (H,W) sigmoid 能量图; gt_bool: (H,W). 返回 per-image 能量统计."""
    halo = binary_dilation(gt_bool, iterations=HALO_DIL) & ~gt_bool
    interior = binary_erosion(gt_bool, iterations=INT_ERODE) & gt_bool
    st = {"gt_px": int(gt_bool.sum()), "halo_px": int(halo.sum()),
          "int_px": int(interior.sum()),
          "E_halo": float(seg[halo].mean()) if halo.sum() else None,
          "E_int": float(seg[interior].mean()) if interior.sum() else None}
    if st["E_halo"] is not None and st["E_int"] is not None:
        st["r"] = st["E_halo"] / max(st["E_int"], EPS)
    else:
        st["r"] = None
    return st


def measure_normal_bg(model, cat, device, bs=16):
    """该类全部正常测试图 seg 输出的全局均值 (E_bg 在线估计, 零标注).

    返回 (E_bg, per_image_means)."""
    dt, gt_t = get_data_transforms(448, 392)
    ds = get_dataset(dataset="MVTec-AD", root=DATA, category=cat, phase="test",
                     data_transform=dt, gt_transform=gt_t, augmentation=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4)
    means = []
    with torch.no_grad():
        for img, gt, label, _ in dl:
            img = img.to(device, non_blocking=True)
            _, seg = run_pass.forward_branches(model, img)
            seg = seg[:, 0].cpu().numpy()
            for i in range(len(label)):
                if int(label[i]) == 0:
                    means.append(float(seg[i].mean()))
    return float(np.mean(means)), means


def measure_real(model, cat, device, bs=16):
    dt, gt_t = get_data_transforms(448, 392)
    ds = get_dataset(dataset="MVTec-AD", root=DATA, category=cat, phase="test",
                     data_transform=dt, gt_transform=gt_t, augmentation=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4)
    recs = []
    with torch.no_grad():
        for img, gt, label, _ in dl:
            img = img.to(device, non_blocking=True)
            _, seg = run_pass.forward_branches(model, img)
            seg = seg[:, 0].cpu().numpy()
            gtn = gt[:, 0].numpy()
            for i in range(len(label)):
                if int(label[i]) != 1:
                    continue
                recs.append(region_stats(seg[i], gtn[i] > 0.5))
    return recs


def measure_synth(model, cat, device, n=N_SYNTH):
    dt, gt_t = get_data_transforms(448, 392)
    ds = get_dataset(dataset="MVTec-AD", root=DATA, category=cat, phase="train",
                     data_transform=dt, gt_transform=gt_t, augmentation=False)
    rng = np.random.RandomState(SEED)
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    recs = []
    with torch.no_grad():
        for i in idxs:
            img_tensor, _, _, _ = ds[i]
            arr = img_tensor.numpy().transpose(1, 2, 0).copy()
            aug, sgt, _ = ds.augment_image_cutpaste(arr)   # 与训练管线同款
            t = torch.from_numpy(aug.transpose(2, 0, 1)).float()[None].to(device)
            _, seg = run_pass.forward_branches(model, t)
            recs.append(region_stats(seg[0, 0].cpu().numpy(), sgt > 0))
    return recs


# ------------------------------------------------------------------- stats

def summarize_s(recs, e_bg):
    """s = E_halo - E_bg 的分布 (s 可为负, 线性空间; 不需要 interior)."""
    ss = np.array([x["E_halo"] - e_bg for x in recs if x["E_halo"] is not None],
                  dtype=np.float64)
    out = {"n": int(len(ss)), "s_values": [float(v) for v in ss]}
    if len(ss):
        out["s_mean"] = float(ss.mean())
        out["s_std"] = float(ss.std())
        out["s_median"] = float(np.median(ss))
        out["s_percentiles"] = {str(p): float(np.percentile(ss, p))
                                for p in (10, 25, 50, 75, 90)}
    return out


def separability_s(real_s, synth_s):
    """线性空间可分性. 约定方向: real 判 s>=tau, synth 判 s<tau.
    同时报告反向最优 (检测方向翻转的类)."""
    a = np.asarray(real_s, dtype=np.float64)
    b = np.asarray(synth_s, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return None
    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    if hi - lo < 1e-9:
        return {"bc": 1.0, "note": "degenerate"}
    bins = np.linspace(lo, hi, 41)
    pa, _ = np.histogram(a, bins=bins, density=True)
    pb, _ = np.histogram(b, bins=bins, density=True)
    pa = pa / pa.sum(); pb = pb / pb.sum()
    bc = float(np.sum(np.sqrt(pa * pb)))
    cand = np.unique(np.concatenate([a, b]))
    best_fwd, best_rev = (None, -1.0), (None, -1.0)
    for t in cand:
        bal_f = 0.5 * ((a >= t).mean() + (b < t).mean())   # real 高
        bal_r = 0.5 * ((a < t).mean() + (b >= t).mean())   # real 低 (翻转)
        if bal_f > best_fwd[1]:
            best_fwd = (float(t), float(bal_f))
        if bal_r > best_rev[1]:
            best_rev = (float(t), float(bal_r))
    flipped = bool(best_rev[1] > best_fwd[1])
    bt = best_rev if flipped else best_fwd
    return {"bc": bc, "flipped": flipped,
            "best_tau": bt[0], "best_bal_acc": bt[1],
            "best_tau_fwd": best_fwd[0], "best_bal_acc_fwd": best_fwd[1],
            "frac_real_s_ge_tau_best": float((a >= bt[0]).mean()) if not flipped
                                       else float((a < bt[0]).mean()),
            "frac_synth_s_correct_side": float((b < bt[0]).mean()) if not flipped
                                         else float((b >= bt[0]).mean())}


# ------------------------------------------------------------------- plot

def plot_s(stats, out_png, tau_ref):
    fig, axes = plt.subplots(4, 4, figsize=(13.5, 11.5))
    for ax, cat in zip(axes.flat, CATS):
        s = stats["halo"][cat]
        rv = np.asarray(s["real"]["s_values"])
        sv = np.asarray(s["synth"]["s_values"])
        parts = ax.violinplot([rv, sv], positions=[0, 1], widths=0.75,
                              showmedians=True, showextrema=False)
        for b, c in zip(parts["bodies"], ["#D55E00", "#0072B2"]):
            b.set_facecolor(c); b.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        sep = s["separability_s"]
        ax.axhline(tau_ref, ls=":", lw=1.0, color="gray")
        flip = " FLIP" if sep["flipped"] else ""
        ax.set_title(f"{cat}  BC={sep['bc']:.2f}{flip}", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["real", "synth"], fontsize=8)
        ax.tick_params(labelsize=8)
    ax = axes.flat[15]
    for pos, src, c in [(0, "real", "#D55E00"), (1, "synth", "#0072B2")]:
        vals = []
        for cat in CATS:
            vals += stats["baseline"][cat][src]["s_values"]
        parts = ax.violinplot([np.asarray(vals)], positions=[pos], widths=0.75,
                              showmedians=True, showextrema=False)
        parts["bodies"][0].set_facecolor(c); parts["bodies"][0].set_alpha(0.75)
        parts["cmedians"].set_color("black")
    ax.axhline(tau_ref, ls=":", lw=1.0, color="gray")
    ax.set_title("baseline (collapsed, pooled)", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["real", "synth"], fontsize=8)
    ax.tick_params(labelsize=8)
    fig.suptitle("G2 gating prior v2:  s = E(halo band) - E_bg(normal images)\n"
                 "halo-arm ckpt per class (15 panels) + collapsed baseline pooled; "
                 f"dotted line = pooled best tau={tau_ref:.3f} (data-derived)",
                 fontsize=11)
    fig.supylabel("s (sigmoid energy units)", x=0.01)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"saved {out_png}")


# ------------------------------------------------------------------- main

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_seed(SEED)
    device = "cuda:0"
    stats = {}
    for arm in ["halo", "baseline"]:
        t0 = time.time()
        print(f"[arm] {arm}", flush=True)
        model = run_pass.build_model(arm, device)
        stats[arm] = {}
        for cat in CATS:
            e_bg, bg_means = measure_normal_bg(model, cat, device)
            real = measure_real(model, cat, device)
            synth = measure_synth(model, cat, device)
            rs, ys = summarize_s(real, e_bg), summarize_s(synth, e_bg)
            sep = separability_s(rs["s_values"], ys["s_values"])
            stats[arm][cat] = {
                "E_bg": e_bg,
                "E_bg_per_image_std": float(np.std(bg_means)),
                "n_normal_bg": len(bg_means),
                "real": rs, "synth": ys, "separability_s": sep,
                "real_r_median": float(np.median(
                    [x["r"] for x in real if x["r"] is not None])),
                "synth_r_median": float(np.median(
                    [x["r"] for x in synth if x["r"] is not None])),
            }
            print(f"[{arm}/{cat}] E_bg={e_bg:.4f} | "
                  f"real s_med={rs.get('s_median', float('nan')):+.4f} "
                  f"synth s_med={ys.get('s_median', float('nan')):+.4f} | "
                  f"BC={sep.get('bc', float('nan')):.3f} "
                  f"flip={sep.get('flipped', 'deg')} "
                  f"tau*={sep.get('best_tau', float('nan')):+.3f} "
                  f"balAcc={sep.get('best_bal_acc', float('nan')):.3f}",
                  flush=True)
        del model; torch.cuda.empty_cache()
        print(f"[arm-done] {arm} ({time.time() - t0:.0f}s)", flush=True)

    # pooled (halo 臂)
    rr = [v for cat in CATS for v in stats["halo"][cat]["real"]["s_values"]]
    ss = [v for cat in CATS for v in stats["halo"][cat]["synth"]["s_values"]]
    pooled = separability_s(rr, ss)
    # Line A 场景判断: 合成异常 s 是否集中在门开侧 (s < tau)
    tau_star = pooled["best_tau"]
    lineA = {"tau_star_pooled": tau_star,
             "frac_synth_gate_open(s<tau)": float((np.asarray(ss) < tau_star).mean()),
             "frac_real_gate_closed(s>=tau)": float((np.asarray(rr) >= tau_star).mean())}
    # T 建议: 过渡带覆盖 synth 分布上尾到 real 分布下尾 -> T = pooled synth std / 2
    t_sugg = float(np.std(ss) / 2)
    out = {
        "experiment": "G2 gating statistic prior v2: s = E_halo - E_bg (SGHL v2)",
        "date": time.strftime("%Y-%m-%d"),
        "s_definition": f"s = mean(seg_sigmoid[halo]) - E_bg(class); halo = "
                        f"dilate(gt,{HALO_DIL})-gt; E_bg = 该类全部正常测试图 "
                        f"seg 输出全局均值 (在线估计, 零标注); s 不需要 interior",
        "gate_convention": "g = sigmoid(-(s-tau)/T); s 低 -> HS 全开 (合成), "
                           "s 高 -> HS 退火 (真实)",
        "arms": {"halo": str(run_pass.CKPTS["halo"]),
                 "baseline": str(run_pass.CKPTS["baseline"])},
        "synth_generator": "MVTecDataset.augment_image_cutpaste (Line A 同款), "
                           f"{N_SYNTH}/class, RandomState({SEED})",
        "seed": SEED,
        "per_arm_per_class": stats,
        "halo_pooled_separability_s": pooled,
        "lineA_gate_check": lineA,
        "suggested_init": {"tau": tau_star, "T": t_sugg,
                           "note": "T = pooled synth s 的 std/2, 使 sigmoid 过渡带 "
                                   "(~±2T) 覆盖 synth 上尾到 real 下尾"},
    }
    jp = OUT_DIR / "gating_prior_s_stats.json"
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {jp}", flush=True)
    print(f"[lineA-check] synth gate-open frac={lineA['frac_synth_gate_open(s<tau)']:.3f} "
          f"real gate-closed frac={lineA['frac_real_gate_closed(s>=tau)']:.3f} "
          f"@tau={tau_star:.3f}, T_sugg={t_sugg:.4f}", flush=True)
    plot_s(stats, OUT_DIR / "fig_gating_prior_s.png", tau_star)
    print("[gating-prior-v2] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
