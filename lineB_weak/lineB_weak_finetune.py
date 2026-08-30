#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Line B -- 弱监督 halo 精修 (k-shot real-GT fine-tune of the Stage-2 seg head).

假设 (Line A 已验证): Dice+HS+HIC 的 halo loss 用 *合成* GT 构造的 halo 带可显著
改善像素分割 (P-F1 0.5096 -> 0.6177)。Line B 验证: 用 *真实* 标注 GT 构造 halo
带做 k-shot 精修, 收益应更大, 并产出 1->5->10-shot 样本效率曲线。

受控双臂 (同一 k / 同一批样本 / 同一精修日程 / 同一起点 ckpt):
  - base : Dice-only (segmentation_loss) 精修
  - halo : Dice + HS + HIC (HaloRefinerLoss, 配置与 Line A 完全一致:
           lambda_hs=1.0, lambda_hic=1.0, halo_radius=5, interior_erode=3,
           hic_margin=0.3, pos_weight=8.0), halo 带由真实 GT 构造。

v2 归因臂 (lambda 置零复用同一 HaloRefinerLoss, 其余与 v1 完全一致):
  - hs_only  : Dice + HS  (lambda_hs=1.0, lambda_hic=0.0)
  - hic_only : Dice + HIC (lambda_hs=0.0, lambda_hic=1.0)
v4 门控臂 (SGHL v2 核心):
  - gated    : Dice + HIC(常数1.0) + g*HS, g = clip(sigmoid((tau_c - s)/T_c), 0.3, 1)
               逐样本自适应; s = E_halo - E_bg (先验测量见 results/gating_prior/),
               tau_c/T_c/E_bg 由 calibrate_gating 用起点 ckpt 在 k-shot 训练样本上
               在线校准 (零标注)。

协议 (DevNet/DRA 标准 k-shot + EVAL_PROTOCOL.md v1):
  1. shot 样本: 每类 test 集的异常图中, 用 RandomState(seed) 固定排列取前 k 张
     (嵌套: k=1 的样本 ⊂ k=5 ⊂ k=10)。抽样清单落盘
     results/lineB/lineB_shot_sampling_seed{seed}.json (--seed 控制, 默认 1)。
     被抽中的异常图从该类最终评估集中剔除, 评估在剩余全部 test 图上进行。
     正常图全部留在评估集中 (shot 只抽异常图)。
  2. 配对正常图: 每类从 train/good 用 RandomState(seed) 取前 k 张 (train 正常图
     本就不在评估总体中, 无泄漏), 与异常图等量, 防止精修后正常区域过火。
     精修数据集 = 15 类 x (k 异常 + k 正常), shuffle 后 batch 混合
     (总量 1:1 平衡; batch 内不强制配比)。
  3. 精修日程 (默认): 只训 Stage-2 seg head (model.freeze_stage1(), 不重初始化,
     双臂同起点 = Line A halo 臂 model_stage2.pth), lr=1e-4, epochs=10,
     batch_size=16 (drop_last=False), StableAdamW + WarmCosine (min lr=1e-5),
     grad clip 0.1 -- 与 Line A stage2 日程同构, 仅 lr 降到 1e-4 适配小样本。
  4. 评估: 完全复用训练管线 evaluate 路径
     (utils.evaluation_batch_with_MaskDecoder, model.inf Eq.9 组合图,
     sigmoid -> 256 双线性 -> gaussian k5 s4 -> gt nearest thr0.5,
     image score = top-1% 像素均值; P-F1max 精确 sklearn
     precision_recall_curve 全像素 -- 与 Line A 主表口径一致;
     protocol_reeval 已证实该路径与精确复评 abs_diff~0)。
     指标: per-class + macro 的 I-AUROC/I-AP/I-F1/P-AUROC/P-AP/P-F1/AUPRO,
     并记录 n_normal/n_anomaly/r_pos。
  5. 产出: results/lineB/{YYYYMMDD}_lineB_inpformer_mvtecad_k{shot}_{arm}_seed{seed}.json
     含 per-category 明细 + 配置 + 抽样清单引用 + 与对臂 (同 seed) 的配对差值 (若对臂
     JSON 已存在则写入)。精修 ckpt 存 lineB_ckpts/k{shot}_{arm}_seed{seed}/model_stage2_ft.pth。
  6. 幂等: 结果 JSON 已存在则跳过 (--force 重跑)。状态追加
     results/lineB_status.txt。

用法 (远程 /data/repo/halo_ad 下):
  # 冒烟: bottle, k=1, 双臂各 2 epoch
  python lineB_weak_finetune.py --categories bottle --k 1 --epochs 2 --tag smoke
  # 完整队列: 3 k 值 x 2 臂 x 15 类
  python lineB_weak_finetune.py --queue
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if not hasattr(np, "trapz"):  # adeval compat shim (numpy>=2)
    np.trapz = np.trapezoid

ROOT = Path("/data/repo/halo_ad")
INP = ROOT / "inp-former-pp"
os.chdir(ROOT)                      # backbones/weights 相对项目根
sys.path.insert(0, str(INP))
sys.path.insert(0, str(ROOT))       # mve_be_cas.halo_losses

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset

import train_pp
from dataset import get_dataset, get_data_transforms
from utils import (setup_seed, segmentation_loss, get_gaussian_kernel,
                   evaluation_batch_with_MaskDecoder)
from mve_be_cas.halo_losses import HaloRefinerLoss, make_region_masks

G_MIN = 0.3  # 门控下界: HS 永不全关 (Line A 合成场景保护)

DATA = "/data/repo/AD-DINOv3/Data/Industrial_Datasets/MVTechAD"
CFG = "dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6"
START_CKPT = ROOT / "saved_results" / f"halo15_{CFG}_halo" / "model_stage2.pth"
CATS = ["carpet", "grid", "leather", "tile", "wood", "bottle", "cable", "capsule",
        "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]
OUT_DIR = ROOT / "results" / "lineB"
STATUS = ROOT / "results" / "lineB_status.txt"
CKPT_DIR = ROOT / "lineB_ckpts"
SEED = 1
METRICS = ["I-AUROC", "I-AP", "I-F1_max", "P-AUROC", "P-AP", "P-F1_max", "AUPRO"]


def log_status(msg):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with open(STATUS, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ------------------------------------------------------------------- sampling

def build_or_load_sampling(cats, kmax=10, seed=SEED):
    """每类 test 异常图的固定随机排列 (seed 可控, 嵌套 k-shot) + train 正常图排列.

    返回 {cat: {"anomaly_order": [相对路径...], "normal_order": [...],
                "n_test_anomaly": int, "n_train_normal": int}}.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpath = OUT_DIR / f"lineB_shot_sampling_seed{seed}.json"
    if mpath.exists():
        with open(mpath) as f:
            manifest = json.load(f)
        # 防御: 类别集合变化时重采
        if set(manifest["categories"].keys()) == set(cats) and \
                manifest.get("kmax", 0) >= kmax:
            return manifest["categories"]
    manifest = {"seed": seed, "kmax": kmax,
                "note": f"anomaly_order: test 集异常图 RandomState({seed}) 排列, 前 k 张为 "
                        f"k-shot 样本 (嵌套); normal_order: train/good RandomState({seed}) 排列, "
                        "前 k 张为配对正常图。",
                "categories": {}}
    for cat in cats:
        dt, gt_t = get_data_transforms(448, 392)
        test = get_dataset(dataset="MVTec-AD", root=DATA, category=cat, phase="test",
                           data_transform=dt, gt_transform=gt_t, augmentation=False)
        train = get_dataset(dataset="MVTec-AD", root=DATA, category=cat, phase="train",
                            data_transform=dt, gt_transform=gt_t, augmentation=False)
        anom_idx = [i for i, lb in enumerate(test.labels) if int(lb) == 1]
        rs = np.random.RandomState(seed)
        anom_order = [str(test.img_paths[i]) for i in
                      np.array(anom_idx)[rs.permutation(len(anom_idx))]]
        norm_idx = list(range(len(train.img_paths)))  # train 全是 good
        rs2 = np.random.RandomState(seed)
        norm_order = [str(train.img_paths[i]) for i in
                      np.array(norm_idx)[rs2.permutation(len(norm_idx))]]
        assert len(anom_order) >= kmax, f"{cat}: 异常图 {len(anom_order)} < kmax={kmax}"
        assert len(norm_order) >= kmax, f"{cat}: 正常图不足"
        manifest["categories"][cat] = {
            "anomaly_order": anom_order, "normal_order": norm_order,
            "n_test_anomaly": len(anom_order), "n_train_normal": len(norm_order)}
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest["categories"]


# ------------------------------------------------------------------- datasets

def build_finetune_dataset(cats, sampling, k):
    """15 类 x (k 张 test 异常 + k 张 train 正常) 的 ConcatDataset."""
    dt, gt_t = get_data_transforms(448, 392)
    parts = []
    for cat in cats:
        test = get_dataset(dataset="MVTec-AD", root=DATA, category=cat, phase="test",
                           data_transform=dt, gt_transform=gt_t, augmentation=False)
        train = get_dataset(dataset="MVTec-AD", root=DATA, category=cat, phase="train",
                            data_transform=dt, gt_transform=gt_t, augmentation=False)
        shot_paths = set(sampling[cat]["anomaly_order"][:k])
        shot_idx = [i for i, p in enumerate(test.img_paths) if str(p) in shot_paths]
        norm_paths = set(sampling[cat]["normal_order"][:k])
        norm_idx = [i for i, p in enumerate(train.img_paths) if str(p) in norm_paths]
        assert len(shot_idx) == k and len(norm_idx) == k, f"{cat}: 抽样解析失败"
        parts.append(Subset(test, shot_idx))
        parts.append(Subset(train, norm_idx))
    return ConcatDataset(parts)


def build_eval_subset(cat, sampling, k):
    """该类 test 全集剔除 k 张 shot 异常图后的 Subset (正常图全保留)."""
    dt, gt_t = get_data_transforms(448, 392)
    test = get_dataset(dataset="MVTec-AD", root=DATA, category=cat, phase="test",
                       data_transform=dt, gt_transform=gt_t, augmentation=False)
    shot_paths = set(sampling[cat]["anomaly_order"][:k])
    keep = [i for i, p in enumerate(test.img_paths) if str(p) not in shot_paths]
    return test, Subset(test, keep)


# ------------------------------------------------------------------- model

def build_model(device):
    args = SimpleNamespace(encoder="dinov2reg_vit_base_14",
                           dinov3_pretrained_dir="backbones/weights",
                           crop_size=392, INP_num=6)
    model, parts = train_pp.build_pp_model(args, device)
    sd = torch.load(START_CKPT, map_location=device)
    model.load_state_dict(sd, strict=True)
    return model, parts


# ------------------------------------------------------------------- gating

def _cat_of(path):
    """.../MVTechAD/{cat}/{train|test}/{defect|good}/xxx.png -> {cat}."""
    return str(path).replace("\\", "/").split("/")[-4]


def calibrate_gating(model, cats, sampling, k, device, print_fn):
    """SGHL v2 门控校准 (起点 ckpt, 精修前在线估计, 零标注):

      s(img) = E_halo(img) - E_bg(class)
      E_bg   = 该类 k 张配对训练正常图的 seg(sigmoid) 输出全局均值
      tau_c  = 该类 k 张训练异常的 s 中位数;  T_c = 该类 s 的 std/2
               (k<2 或 std<1e-6 时 fallback: 全部训练异常 pooled 的 std/2)
      g      = clip(sigmoid((tau_c - s)/T_c), G_MIN, 1), 逐样本乘在 HS 上.

    返回 {cat: {"tau","T","E_bg","s_values","mean_g_at_start"}}.
    """
    dt, gt_t = get_data_transforms(448, 392)
    gate = {}
    model.eval()
    with torch.no_grad():
        for cat in cats:
            test = get_dataset(dataset="MVTec-AD", root=DATA, category=cat,
                               phase="test", data_transform=dt, gt_transform=gt_t,
                               augmentation=False)
            train = get_dataset(dataset="MVTec-AD", root=DATA, category=cat,
                                phase="train", data_transform=dt, gt_transform=gt_t,
                                augmentation=False)
            shot_paths = set(sampling[cat]["anomaly_order"][:k])
            norm_paths = set(sampling[cat]["normal_order"][:k])
            shot_idx = [i for i, p in enumerate(test.img_paths)
                        if str(p) in shot_paths]
            norm_idx = [i for i, p in enumerate(train.img_paths)
                        if str(p) in norm_paths]
            # E_bg: k 张配对正常图的 seg 输出全局均值
            imgs_n = torch.stack([train[i][0] for i in norm_idx]).to(device)
            _, _, _, m = model.forward_stage2(imgs_n)
            e_bg = float(torch.sigmoid(m).mean())
            # 训练异常逐图 s
            s_vals = []
            imgs_a = torch.stack([test[i][0] for i in shot_idx]).to(device)
            gts_a = torch.stack([test[i][1] for i in shot_idx]).to(device)
            _, _, _, m = model.forward_stage2(imgs_a)
            prob = torch.sigmoid(m)
            halo_m, _, _ = make_region_masks((gts_a > 0).float(), 5, 3)
            e_h = ((prob * halo_m).sum(dim=(2, 3))
                   / halo_m.sum(dim=(2, 3)).clamp(min=1.0)).squeeze(-1)
            s_vals = [float(v) for v in (e_h - e_bg).cpu().numpy()]
            gate[cat] = {"E_bg": e_bg, "s_values": s_vals}
    # tau / T (含 pooled fallback)
    pooled_s = [v for c in cats for v in gate[c]["s_values"]]
    t_pool = max(float(np.std(pooled_s)) / 2.0, 1e-3)
    for cat in cats:
        sv = np.asarray(gate[cat]["s_values"], dtype=np.float64)
        tau = float(np.median(sv))
        T = float(np.std(sv) / 2.0)
        fb = bool(len(sv) < 2 or T < 1e-6)
        if fb:
            T = t_pool
        gate[cat].update({"tau": tau, "T": T, "T_fallback": fb,
                          "mean_g_at_start": float(np.mean(np.clip(
                              1.0 / (1.0 + np.exp(-(tau - sv) / T)),
                              G_MIN, 1.0)))})
    print_fn(f"[gate] calibrated: tau range "
             f"[{min(g['tau'] for g in gate.values()):+.3f}, "
             f"{max(g['tau'] for g in gate.values()):+.3f}], "
             f"T_pool={t_pool:.4f}, "
             f"mean_g_at_start range [{min(g['mean_g_at_start'] for g in gate.values()):.2f}, "
             f"{max(g['mean_g_at_start'] for g in gate.values()):.2f}]")
    return gate


# ------------------------------------------------------------------- train

def finetune(model, parts, ft_data, arm, epochs, lr, batch_size, device, print_fn,
             gate=None):
    """只训 seg head (freeze_stage1, 不重初始化 -- 各臂同起点精修).

    base: segmentation_loss (Dice). halo: Dice + HaloRefinerLoss.total
    (与 Line A stage2 完全同构: loss = Dice + (BCE+Dice+HS+HIC)).
    v2 归因: hs_only / hic_only 用 lambda 置零复用同一 loss (其余不变).
    gated: Dice + HIC(常数1.0) + g*HS, g=clip(sigmoid((tau_c-s)/T_c),0.3,1)
           逐样本自适应 (s 用当前头的 detach 输出; tau_c/T_c/E_bg 来自
           calibrate_gating 的起点 ckpt 校准).
    """
    model.train()
    model.freeze_stage1()
    seg_head = parts["Seg_Head"]
    assert all(p.requires_grad for p in seg_head.parameters())

    ARM_LAMBDAS = {"halo": (1.0, 1.0), "hs_only": (1.0, 0.0),
                   "hic_only": (0.0, 1.0), "gated": (0.0, 1.0)}
    halo_lossfn = None
    if arm in ARM_LAMBDAS:
        l_hs, l_hic = ARM_LAMBDAS[arm]
        halo_lossfn = HaloRefinerLoss(
            lambda_hs=l_hs, lambda_hic=l_hic, halo_radius=5, interior_erode=3,
            hic_margin=0.3, pos_weight=8.0).to(device)
    if arm == "gated":
        assert gate is not None, "gated 臂需要 calibrate_gating 的校准结果"

    loader = DataLoader(ft_data, batch_size=batch_size, shuffle=True,
                        num_workers=4, drop_last=False)
    opt, sched = train_pp.make_optimizer(
        seg_head.parameters(), lr=lr, total_iters=max(1, epochs * len(loader)))
    print_fn(f"[FT:{arm}] images={len(ft_data)} iters/ep={len(loader)} "
             f"epochs={epochs} lr={lr}")

    nan_guard = False
    for ep in range(epochs):
        losses, hs_l, hic_l, g_l = [], [], [], []
        for img, gt, label, paths in loader:
            img = img.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            _, _, _, mask = model.forward_stage2(img)
            mask = F.interpolate(mask, size=gt.shape[-2:], mode="bilinear",
                                 align_corners=False)
            loss = segmentation_loss(mask=mask, gt=gt)
            if halo_lossfn is not None:
                gt_bin = (gt > 0).float()  # 与 Line A 相同的二值化
                out = halo_lossfn(mask, gt_bin)
                loss = loss + out["total"]
                hs_l.append(float(out["hs"])); hic_l.append(float(out["hic"]))
            if arm == "gated":
                # 逐样本门控 HS (覆盖在 halo_lossfn 的 hs=0 之上)
                prob = torch.sigmoid(mask)
                halo_m, _, _ = make_region_masks(gt_bin, 5, 3)
                denom = halo_m.sum(dim=(2, 3)).clamp(min=1.0)
                hs_per = ((prob * halo_m).sum(dim=(2, 3)) / denom).squeeze(-1)
                has = (halo_m.sum(dim=(2, 3)) > 0).float().squeeze(-1)
                tau_t = torch.tensor([gate[_cat_of(p)]["tau"] for p in paths],
                                     dtype=torch.float32, device=device)
                T_t = torch.tensor([gate[_cat_of(p)]["T"] for p in paths],
                                   dtype=torch.float32, device=device)
                ebg_t = torch.tensor([gate[_cat_of(p)]["E_bg"] for p in paths],
                                     dtype=torch.float32, device=device)
                with torch.no_grad():
                    g = torch.clamp(torch.sigmoid((tau_t - (hs_per - ebg_t)) / T_t),
                                    min=G_MIN, max=1.0)
                hs = (hs_per * g * has).sum() / has.sum().clamp(min=1.0)
                loss = loss + hs
                g_l.append(float((g * has).sum() / has.sum().clamp(min=1.0)))
            if not torch.isfinite(loss):
                print_fn(f"[FT:{arm}] NaN/Inf loss @ epoch {ep + 1}, ABORT")
                nan_guard = True
                break
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(seg_head.parameters(), max_norm=0.1)
            opt.step(); sched.step()
            losses.append(float(loss))
        if nan_guard:
            break
        msg = f"[FT:{arm}] epoch [{ep + 1}/{epochs}] loss={np.mean(losses):.4f}"
        if halo_lossfn is not None:
            msg += f" hs={np.mean(hs_l):.4f} hic={np.mean(hic_l):.4f}"
        if arm == "gated":
            msg += f" mean_g={np.mean(g_l):.3f}"
        print_fn(msg)
    return nan_guard


# ------------------------------------------------------------------- eval

def evaluate(model, cats, sampling, k, device, batch_size, print_fn):
    """按 Line A 口径评估剔除 shot 后的测试集; 返回 per-category 明细."""
    per_cat = {}
    for cat in cats:
        test, sub = build_eval_subset(cat, sampling, k)
        dl = DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=4)
        res = evaluation_batch_with_MaskDecoder(
            model, dl, device, max_ratio=0.01, resize_mask=256)
        # n_normal / n_anomaly / r_pos (r_pos 在 256^2 nearest GT 上统计, 与 Line A 一致)
        labels = np.array([int(test.labels[i]) for i in sub.indices])
        n_normal = int((labels == 0).sum()); n_anom = int((labels == 1).sum())
        pos = tot = 0
        for i in sub.indices:
            _, gt, _, _ = test[i]
            g = F.interpolate(gt[None], size=(256, 256), mode="nearest")
            pos += int((g > 0.5).sum()); tot += g.numel()
        per_cat[cat] = {m: float(v) for m, v in zip(METRICS, res[:7])}
        per_cat[cat]["P-DICE"] = float(res[7])
        per_cat[cat]["n_normal"] = n_normal
        per_cat[cat]["n_anomaly"] = n_anom
        per_cat[cat]["r_pos"] = pos / max(1, tot)
        print_fn(f"[Eval] {cat}: I-AUROC={per_cat[cat]['I-AUROC']:.4f} "
                 f"P-AUROC={per_cat[cat]['P-AUROC']:.4f} "
                 f"P-AP={per_cat[cat]['P-AP']:.4f} P-F1={per_cat[cat]['P-F1_max']:.4f} "
                 f"(n_normal={n_normal} n_anomaly={n_anom})")
    return per_cat


def result_path(k, arm, tag=None, seed=SEED):
    date = datetime.now().strftime("%Y%m%d")
    name = f"{date}_lineB_inpformer_mvtecad_k{k}_{arm}_seed{seed}.json"
    if tag:
        name = name.replace(".json", f"_{tag}.json")
    return OUT_DIR / name


def paired_diff(k, arm, per_cat, tag=None, seed=SEED):
    """若对臂同 k 同 seed 的 JSON 已存在, 计算 per-category 配对差值."""
    other = "halo" if arm == "base" else "base"
    op = result_path(k, other, tag, seed)
    if not op.exists():
        return None
    with open(op) as f:
        odat = json.load(f)
    oper = odat["per_category"]
    rows = []
    for cat, m in per_cat.items():
        if cat not in oper:
            continue
        row = {"cat": cat}
        for met in METRICS:
            row[f"d_{met}"] = m[met] - oper[cat][met]
        rows.append(row)
    return {"vs": other, "convention": f"{arm} - {other}",
            "per_category": rows,
            "macro": {f"d_{met}": float(np.mean([r[f"d_{met}"] for r in rows]))
                      for met in METRICS}}


# ------------------------------------------------------------------- one run

def run_one(k, arm, cats, epochs, lr, batch_size, device, tag=None, force=False,
            seed=SEED):
    op = result_path(k, arm, tag, seed)
    if op.exists() and not force:
        log_status(f"[skip] k={k} arm={arm} seed={seed} exists: {op.name}")
        return True
    t0 = time.time()
    log_status(f"[start] k={k} arm={arm} seed={seed} cats={len(cats)} "
               f"epochs={epochs} lr={lr} tag={tag}")
    setup_seed(seed)
    sampling = build_or_load_sampling(cats, kmax=max(10, k), seed=seed)
    ft_data = build_finetune_dataset(cats, sampling, k)
    model, parts = build_model(device)

    gate = None
    if arm == "gated":
        gate = calibrate_gating(model, cats, sampling, k, device, log_status)
        model.train()

    nan = finetune(model, parts, ft_data, arm, epochs, lr, batch_size, device,
                   print_fn=log_status, gate=gate)
    if nan:
        log_status(f"[FAIL-NaN] k={k} arm={arm} seed={seed}")
        del model; torch.cuda.empty_cache()
        return False

    ck_dir = CKPT_DIR / f"k{k}_{arm}_seed{seed}"
    ck_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ck_dir / "model_stage2_ft.pth")

    model.eval()
    per_cat = evaluate(model, cats, sampling, k, device, batch_size, log_status)
    macro = {m: float(np.mean([per_cat[c][m] for c in cats])) for m in METRICS}

    out = {
        "protocol": (f"EVAL_PROTOCOL.md v1 + DevNet/DRA k-shot: 每类 test 异常图 "
                     f"RandomState({seed}) 嵌套抽样, shot 图从评估集剔除; 评估口径 = "
                     "训练管线 evaluation_batch_with_MaskDecoder (model.inf Eq.9, "
                     "P-F1max 精确 sklearn PR 全像素 @256)"),
        "line": "B", "k_shot": k, "arm": arm, "seed": seed,
        "config": CFG,
        "script": "lineB_weak_finetune.py (2026-08-13, v4: +gated 门控 HS 臂)",
        "start_ckpt": str(START_CKPT),
        "finetune_schedule": {
            "trainable": "seg_head only (freeze_stage1, 不重初始化, 各臂同起点)",
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "optimizer": "StableAdamW + WarmCosine(final=0.1*lr), clip 0.1",
            "loss": {"base": "segmentation_loss (Dice)",
                     "halo": "Dice + HaloRefinerLoss(BCE+Dice+HS+HIC, hs=1, hic=1, "
                             "dilate=5, erode=3, margin=0.3, pos_weight=8, 真实GT构造halo带)",
                     "hs_only": "Dice + HaloRefinerLoss(hs=1, hic=0, 其余同 halo)",
                     "hic_only": "Dice + HaloRefinerLoss(hs=0, hic=1, 其余同 halo)",
                     "gated": "Dice + HIC(常数1.0) + g*HS (lambda_hs=1), "
                              "g=clip(sigmoid((tau_c-s)/T_c), 0.3, 1) 逐样本; "
                              "s=E_halo-E_bg(该类k张配对正常图seg均值); "
                              "tau_c=k-shot训练异常s中位数, T_c=s.std/2 (pooled fallback); "
                              "校准用起点ckpt, 训练中s用当前头detach输出"}[arm],
            "normal_pairing": f"每类 train/good RandomState({seed}) 前 k 张 (等量配对)"},
        "sampling_manifest": str(OUT_DIR / f"lineB_shot_sampling_seed{seed}.json"),
        "shots_used": {c: sampling[c]["anomaly_order"][:k] for c in cats},
        "categories": cats, "n_categories": len(cats),
        "per_category": per_cat, "macro": macro,
        "finetune_ckpt": str(ck_dir / "model_stage2_ft.pth"),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    if arm == "gated" and gate is not None:
        out["gating"] = {
            "formula": f"g = clip(sigmoid((tau_c - s)/T_c), {G_MIN}, 1); "
                       "s = E_halo(img) - E_bg(class)",
            "g_min": G_MIN,
            "per_class": {c: {"tau": gate[c]["tau"], "T": gate[c]["T"],
                              "E_bg": gate[c]["E_bg"],
                              "T_fallback": gate[c]["T_fallback"],
                              "s_values_kshot": gate[c]["s_values"],
                              "mean_g_at_start": gate[c]["mean_g_at_start"]}
                          for c in cats},
            "mean_g_at_start_macro": float(np.mean(
                [gate[c]["mean_g_at_start"] for c in cats])),
        }
        out["adjudication_note"] = (
            "预定义裁决标准: (1) gated I-AUROC 相对 base 损伤 <= 0.005 "
            "(修复 halo 臂 -0.022 病灶); (2) gated P-F1/P-AP 增益不低于 hic_only "
            "增益幅度的 80%; (3) mean g < 0.9 (门控在真实数据上确实部分关闭且不崩).")
    pd = paired_diff(k, arm, per_cat, tag, seed)
    if pd is not None:
        out["paired_diff"] = pd
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2)
    log_status(f"[done] k={k} arm={arm} seed={seed} P-F1_macro={macro['P-F1_max']:.4f} "
               f"P-AP_macro={macro['P-AP']:.4f} I-AUROC_macro={macro['I-AUROC']:.4f} "
               f"-> {op.name} ({out['elapsed_sec']}s)")
    del model; torch.cuda.empty_cache()
    return True


def main():
    ap = argparse.ArgumentParser(description="Line B weakly-supervised halo fine-tune")
    ap.add_argument("--k", type=int, nargs="*", default=[1, 5, 10])
    ap.add_argument("--arms", nargs="*", default=["base", "halo"],
                    choices=["base", "halo", "hs_only", "hic_only", "gated"])
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--queue", action="store_true",
                    help="完整队列模式: 3 k x 2 臂串行, 状态写 lineB_status.txt")
    ap.add_argument("--tag", type=str, default=None,
                    help="结果 JSON 追加标签 (如 smoke), 避免与正式结果混淆")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="shot 抽样与精修内部随机性的种子 (默认 1); 结果 JSON 与抽样 "
                         "清单文件名均带 seed")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    assert START_CKPT.exists(), f"起点 ckpt 不存在: {START_CKPT}"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cats = args.categories or CATS
    ks = args.k if not args.queue else [1, 5, 10]
    arms = args.arms
    tag = args.tag if not args.queue else None

    ok_all = True
    for k in ks:
        for arm in arms:
            ok = run_one(k, arm, cats, args.epochs, args.lr, args.batch_size,
                         device, tag=tag, force=args.force, seed=args.seed)
            ok_all = ok_all and ok
    log_status(f"[queue-complete] ks={ks} arms={arms} seed={args.seed} ok={ok_all}")
    if not ok_all:
        sys.exit(2)


if __name__ == "__main__":
    main()
