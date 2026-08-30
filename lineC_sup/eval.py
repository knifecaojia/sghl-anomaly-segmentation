#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Line C 评估 (EVAL_PROTOCOL.md v1 约束).

指标 (per category, 全量官方测试集, 不做正常图下采样):
  P-AUROC / P-AP (sklearn, 全像素) / P-F1max (精确全像素排序; 像素数 >1.5e8 时按协议
  允许的 >=1000 档阈值扫描, 用 4096 档并在 JSON 标注 f1_method) /
  I-AUROC (图级分数 = 像素 max) / AUPRO (MVTec 官方: 连通域 PRO, FPR<=0.3, 200 阈值).
汇总: macro 均值 + bootstrap 95% CI (以类为重采样单位, B=10000).
JSON: results/{YYYYMMDD}_lineC_frozendino_{train集合}_to_{test集}_{arm}_seed{seed}.json
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets as D
from models import LineCModel

LINE_C_DIR = os.path.dirname(os.path.abspath(__file__))
EXACT_PIXEL_LIMIT = int(1.5e8)
N_BINS = 4096
AUPRO_FPR_LIMIT = 0.3
AUPRO_N_THRESH = 200
BOOTSTRAP_B = 10000


def script_version():
    h = hashlib.sha1()
    for f in ('datasets.py', 'models.py', 'losses.py', 'train.py', 'eval.py'):
        p = os.path.join(LINE_C_DIR, f)
        h.update(f.encode())
        h.update(open(p, 'rb').read() if os.path.exists(p) else b'')
    return h.hexdigest()[:12]


@torch.no_grad()
def predict_scores(model, loader):
    """返回 per-sample: prob (H,W) float32 numpy, mask (H,W) uint8, label."""
    out = []
    for batch in loader:
        image = batch['image'].cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            feats = model.extract(image)
            logits = model.head(feats)
        prob = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy()
        prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)  # 塌缩头 NaN 防护
        mask = batch['mask'].squeeze(1).numpy().astype(np.uint8)
        for i in range(len(prob)):
            out.append((prob[i], mask[i], int(batch['label'][i])))
    return out


def exact_pixel_metrics(scores, labels):
    """全像素排序的 F1max/AUROC/AP. scores float32, labels uint8, 均为 1D."""
    order = np.argsort(-scores, kind='mergesort')
    lab = labels[order]
    tp = np.cumsum(lab)
    fp = np.cumsum(1 - lab)
    total_pos = lab.sum()
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(total_pos, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    f1max = float(f1.max())
    auroc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    return f1max, auroc, ap


def binned_pixel_metrics_from_hist(pos_hist, neg_hist):
    """由 4096 档直方图计算 F1max/AUROC/AP (协议允许的 >=1000 档路径)."""
    tp = pos_hist[::-1].cumsum()[::-1]
    fp = neg_hist[::-1].cumsum()[::-1]
    total_pos = pos_hist.sum()
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(total_pos, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    f1max = float(f1.max())
    cum_neg_below = neg_hist.cumsum() - neg_hist  # 严格小于
    auroc = float((pos_hist * (cum_neg_below + 0.5 * neg_hist)).sum()
                  / max(total_pos * neg_hist.sum(), 1))
    rec_next = np.concatenate([rec[1:], [0.0]])
    # AP = Σ_i prec[i]·(rec[i] − rec[i+1]) (沿阈值降低、recall 递增方向积分;
    # 2026-08-15 BUGFIX: 此前误用 rec[i−1] 导致符号反转, 大类直方图路径 AP 出现负值)
    ap = float((prec * (rec - rec_next)).sum())
    return f1max, auroc, ap


def binned_pixel_metrics(scores, labels, n_bins=N_BINS):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(scores, bins) - 1, 0, n_bins - 1)
    pos_hist = np.bincount(idx[labels == 1], minlength=n_bins).astype(np.float64)
    neg_hist = np.bincount(idx[labels == 0], minlength=n_bins).astype(np.float64)
    return binned_pixel_metrics_from_hist(pos_hist, neg_hist)


def compute_aupro(maps_and_masks):
    """AUPRO, 与 MVTec 官方实现 (CostFilter utils.compute_pro_original) 逐条对齐:

      thresholds = np.arange(min, max, (max-min)/200)  -- 全部测试图 amaps 的 min/max
      二值化: amap > th (严格大于)
      PRO(t) = 全部 GT 连通域的像素覆盖率均值 (正常图无连通域, 不贡献)
      FPR(t) = #(负像素 & amap>th) / #负像素, 负像素含**正常图全部像素**
      归一化 (官方关键 quirk): 滤 fpr<0.3 后, fpr 轴除以 max(fpr) 重标定到 [0,1],
      再对 (fpr, pro) 求 auc —— 不是除以固定 0.3!

    [2026-08-15 BUGFIX] 此前版本: (a) 只把异常图计入负像素; (b) 积分除以固定 0.3
    不做 FPR 重标定 —— 对双峰分数分布 (背景~0, 缺陷~1), 线性阈值在 FPR<=0.3 区间
    只覆盖极窄一段, 导致 AUPRO 假性≈0 (bottle hic: 0.0127 vs 手工 PRO(t*)=0.58).
    向量化实现: 每图 digitize(阈值) + bincount, 数学与官方循环版一致 (已数值校验).
    maps_and_masks: list of (prob (H,W) float32, gt (H,W) uint8), **全部测试图**.
    """
    if not maps_and_masks:
        return 0.0
    all_min = min(m.min() for m, _ in maps_and_masks)
    all_max = max(m.max() for m, _ in maps_and_masks)
    if all_max <= all_min:
        return 0.0
    delta = (all_max - all_min) / AUPRO_N_THRESH
    thresholds = np.arange(all_min, all_max, delta)   # 与官方 np.arange 一致
    n_th = len(thresholds)
    nb = n_th + 1
    pro_sum = np.zeros(n_th)
    n_comps = 0
    fp_hist = np.zeros(nb)
    total_neg = 0
    for amap, gt in maps_and_masks:
        # digitize right=True: s > thresholds[i]  <=>  b > i  (与官方 amap>th 一致)
        b = np.digitize(amap, thresholds, right=True).ravel()    # 0..n_th
        lab, n = ndimage.label(gt > 0)
        lab_r = lab.ravel()
        pos = lab_r > 0
        if n > 0:
            joint = np.bincount(lab_r[pos] * nb + b[pos],
                                minlength=(n + 1) * nb).reshape(n + 1, nb)
            sizes = joint.sum(axis=1)
            hits = joint[:, ::-1].cumsum(axis=1)[:, ::-1][:, 1:]  # hits_c[i]=#(b>i)
            pro_sum += (hits[1:] / np.maximum(sizes[1:, None], 1)).sum(axis=0)
            n_comps += n
        fp_hist += np.bincount(b[~pos], minlength=nb)
        total_neg += int((~pos).sum())
    if n_comps == 0 or total_neg == 0:
        return 0.0
    pros = pro_sum / n_comps
    fps = fp_hist[::-1].cumsum()[::-1][1:]                  # fp(t_i) = #(b > i)
    fprs = fps / total_neg
    keep = fprs < AUPRO_FPR_LIMIT                            # 官方: 严格 <
    if keep.sum() < 2:
        return 0.0
    fprs_k = fprs[keep]
    pros_k = pros[keep]
    fpr_max = fprs_k.max()
    if fpr_max <= 0:
        return float(pros_k.mean())                          # 官方会除零; 完美分离边界
    fprs_k = fprs_k / fpr_max                                # 官方重标定到 [0,1]
    order = np.argsort(fprs_k)                               # 升序后梯形积分
    return float(np.trapezoid(pros_k[order], fprs_k[order]))


def bootstrap_ci(values, B=BOOTSTRAP_B, seed=0):
    """以类为重采样单位的 macro 均值 95% CI."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(values.mean()), float(values.mean()), float(values.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(B, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_sets', nargs='+', required=True)
    ap.add_argument('--test_set', required=True)
    ap.add_argument('--arm', required=True,
                    choices=['base', 'sghl', 'hic', 'base_pw', 'hic_adp', 'focal',
                    'hic_m01', 'hic_m05', 'hic_r3', 'hic_r8'])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--max_categories', type=int, default=0,
                    help='>0 时只评估前 N 个类 (冒烟用)')
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()

    tag = f'{"+".join(args.train_sets)}_to_{args.test_set}_{args.arm}_seed{args.seed}'
    ckpt = args.ckpt or os.path.join(LINE_C_DIR, 'checkpoints', tag, 'last.pth')
    date = time.strftime('%Y%m%d')
    out_json = os.path.join(LINE_C_DIR, 'results',
                            f'{date}_lineC_frozendino_{tag}.json')
    print(f'[eval] tag={tag} ckpt={ckpt}', flush=True)

    state = torch.load(ckpt, map_location='cuda')
    model = LineCModel('cuda')
    model.head.load_state_dict(state['head'])
    model.head.eval()

    samples = D.build_test_set(args.test_set)
    cats = sorted({s['category'] for s in samples})
    if args.max_categories:
        cats = cats[:args.max_categories]
        print(f'[eval] SUBSET: only {len(cats)} categories (smoke)', flush=True)
    print(f'[eval] test_set={args.test_set} categories={len(cats)} '
          f'total_samples={sum(1 for s in samples if s["category"] in cats)}', flush=True)

    per_category = {}
    for ci, cat in enumerate(cats):
        t0 = time.time()
        cat_samples = [s for s in samples if s['category'] == cat]
        ds = D.LineCDataset(cat_samples)
        loader = DataLoader(ds, batch_size=args.bs, shuffle=False,
                            num_workers=args.num_workers, collate_fn=D.collate_linec,
                            pin_memory=True)
        preds = predict_scores(model, loader)
        n_normal = sum(1 for _, _, l in preds if l == 0)
        n_anomaly = sum(1 for _, _, l in preds if l == 1)
        n_pixels = sum(p.size for p, _, _ in preds)

        if n_pixels > EXACT_PIXEL_LIMIT:
            # 大类流式 4096 档直方图 (协议允许 >=1000 档; 避免拼接数百 M 像素数组)
            f1_method = f'bins{N_BINS}'
            bins = np.linspace(0.0, 1.0, N_BINS + 1)
            pos_hist = np.zeros(N_BINS)
            neg_hist = np.zeros(N_BINS)
            n_pos = 0
            for p, m, _ in preds:
                idx = np.clip(np.digitize(p.ravel(), bins) - 1, 0, N_BINS - 1)
                lab = m.ravel()
                pos_hist += np.bincount(idx[lab == 1], minlength=N_BINS)
                neg_hist += np.bincount(idx[lab == 0], minlength=N_BINS)
                n_pos += int(lab.sum())
            r_pos = n_pos / n_pixels
            f1max, p_auroc, p_ap = binned_pixel_metrics_from_hist(pos_hist, neg_hist)
        else:
            f1_method = 'exact_sort'
            scores = np.concatenate([p.ravel() for p, _, _ in preds]).astype(np.float32)
            labels = np.concatenate([m.ravel() for _, m, _ in preds]).astype(np.uint8)
            r_pos = float(labels.mean())
            f1max, p_auroc, p_ap = exact_pixel_metrics(scores, labels)
            del scores, labels

        img_scores = np.array([p.max() for p, _, _ in preds])
        img_labels = np.array([l for _, _, l in preds])
        i_auroc = float(roc_auc_score(img_labels, img_scores)) \
            if len(set(img_labels)) > 1 else float('nan')

        aupro = compute_aupro([(p, m) for p, m, _ in preds])  # 官方口径: 全部测试图

        per_category[cat] = dict(
            n_normal=n_normal, n_anomaly=n_anomaly, r_pos=r_pos,
            p_f1max=f1max, p_auroc=p_auroc, p_ap=p_ap,
            i_auroc=i_auroc, aupro=aupro, f1_method=f1_method)
        print(f'[eval] ({ci+1}/{len(cats)}) {cat}: F1max={f1max:.4f} AP={p_ap:.4f} '
              f'P-AUC={p_auroc:.4f} I-AUC={i_auroc:.4f} AUPRO={aupro:.4f} '
              f'({time.time()-t0:.0f}s, n={len(preds)}, r_pos={r_pos:.5f})', flush=True)
        del preds

    metrics = ['p_f1max', 'p_auroc', 'p_ap', 'i_auroc', 'aupro']
    macro = {}
    for m in metrics:
        vals = [per_category[c][m] for c in per_category
                if not (isinstance(per_category[c][m], float)
                        and np.isnan(per_category[c][m]))]
        mean, lo, hi = bootstrap_ci(vals)
        macro[m] = dict(mean=mean, ci95=[lo, hi])

    result = dict(
        script_version=script_version(),
        pipeline='lineC_frozendino', train_sets=args.train_sets,
        test_set=args.test_set, arm=args.arm, seed=args.seed,
        epochs=state.get('args', {}).get('epochs'),
        ckpt_epoch=state.get('epoch'), ckpt=ckpt,
        n_categories=len(per_category),
        subset=bool(args.max_categories),
        eval_protocol='EVAL_PROTOCOL.md v1: F1max exact-sort (bins4096 if >1.5e8 px), '
                      'AUPRO FPR<=0.3, bootstrap B=10000',
        macro=macro, per_category=per_category)
    if args.arm == 'base_pw':
        result['note'] = ('base_pw = 不平衡修正后的公平对照: 与 base 同为 Dice+BCE, '
                          '唯一差异是 BCE 正样本权重 pos_weight=10 (base 臂在真实 GT '
                          'pooled 训练下塌缩为常数输出, 诊断见 20260814 记录); '
                          '论文主消融为 base_pw vs hic, sghl (常数 HS=1.0) 为反面对照.')
    elif args.arm == 'focal':
        result['note'] = ('focal = Dice + Binary Focal(gamma=2, alpha=0.75), 无 BCE/HIC/HS '
                          '(REVIEW_V1 A2 对照: 幅度类重加权的第三种代表, 检验其能否对抗塌缩; '
                          'alpha=0.75 给正类 3x 权重使 focal 拥有不低于 base_pw 的正向强调).')
    elif args.arm in ('hic_m01', 'hic_m05', 'hic_r3', 'hic_r8'):
        _v = dict(hic_m01='margin=0.1', hic_m05='margin=0.5',
                  hic_r3='halo_radius=3', hic_r8='halo_radius=8')
        result['note'] = ('A5 sensitivity variant of hic: ' + _v[args.arm] +
                          ' (其余与 hic 完全一致; OFAT 自基线 m=0.3/r=5).')
    elif args.arm == 'hic_adp':
        result['note'] = ('hic_adp = 自适应形态学 HIC (Real-IAD 小缺陷修复): '
                          'GT 面积 A>=1568 (=32*(2*3+1)^2) 时 erode=3, 否则 erode=1; '
                          'halo 半径 min(5, max(2, int(sqrt(A)/8))); '
                          'A<32 (或 interior/halo 为空) 的样本 HIC 权重置 0. '
                          '动机: 固定 erode=3/halo=5 在小缺陷上 interior 消失、'
                          'halo 带≈缺陷本体, HIC 误压缺陷本体.')
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'[eval] saved {out_json}', flush=True)
    for m in metrics:
        print(f'[eval] macro {m}: {macro[m]["mean"]:.4f} '
              f'CI95=[{macro[m]["ci95"][0]:.4f}, {macro[m]["ci95"][1]:.4f}]', flush=True)


if __name__ == '__main__':
    main()
