#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""bins4096 评测近似抽样验证 (REVIEW_V1 A8 / 协议风险第 3 条).

设计: 对 Real-IAD 前 3 类各取前 700 张测试图 (shovel=False 确定性子集,
~140 Mpx < EXACT_PIXEL_LIMIT=1.5e8, 天然落入精确路径), 同一批预测
分别用 (a) exact 全像素排序 与 (b) 协议 4096 档直方图 计算像素级指标,
差值 = 量化误差本身. AUPRO 对阈值不敏感于 1/4096 量化, 不在本审计范围
(协议表格中 AUPRO 由连续 prob 计算, 不经直方图).

输出: results/bins_audit/audit.json + stdout 对照表.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import datasets as D
from eval import predict_scores, exact_pixel_metrics, binned_pixel_metrics
from models import LineCModel

CKPT = os.path.join(HERE, 'checkpoints', 'mvtec+visa_to_realiad_hic_seed1', 'last.pth')
N_CATS, N_IMGS = 5, 700
OUT = os.path.join(HERE, 'results', 'bins_audit', 'audit.json')


def main():
    state = torch.load(CKPT, map_location='cuda')
    model = LineCModel('cuda')
    model.head.load_state_dict(state['head'])
    model.head.eval()

    samples = D.build_test_set('realiad')
    cats = sorted({s['category'] for s in samples})[:N_CATS]

    rows = []
    for cat in cats:
        t0 = time.time()
        cat_samples = [s for s in samples if s['category'] == cat][:N_IMGS]
        ds = D.LineCDataset(cat_samples)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4,
                            collate_fn=D.collate_linec, pin_memory=True)
        preds = predict_scores(model, loader)
        scores = np.concatenate([p.ravel() for p, _, _ in preds]).astype(np.float32)
        labels = np.concatenate([m.ravel() for _, m, _ in preds]).astype(np.uint8)
        n_px = scores.size
        assert n_px <= int(1.5e8), f'{cat}: {n_px} px exceeds exact limit'

        ex = exact_pixel_metrics(scores, labels)
        bn = binned_pixel_metrics(scores, labels)
        row = dict(
            category=cat, n_images=len(preds), n_pixels=int(n_px),
            r_pos=float(labels.mean()),
            exact=dict(f1max=ex[0], p_auroc=ex[1], p_ap=ex[2]),
            binned=dict(f1max=bn[0], p_auroc=bn[1], p_ap=bn[2]),
            delta=dict(f1max=bn[0] - ex[0], p_auroc=bn[1] - ex[1], p_ap=bn[2] - ex[2]),
            minutes=round((time.time() - t0) / 60, 1))
        rows.append(row)
        print(f"[audit] {cat}: n={len(preds)} imgs, {n_px/1e6:.0f} Mpx, "
              f"F1 exact={ex[0]:.4f} binned={bn[0]:.4f} d={bn[0]-ex[0]:+.4f} | "
              f"AUROC d={bn[1]-ex[1]:+.4f} | AP d={bn[2]-ex[2]:+.4f} "
              f"({row['minutes']}min)", flush=True)

    d_f1 = [abs(r['delta']['f1max']) for r in rows]
    d_au = [abs(r['delta']['p_auroc']) for r in rows]
    d_ap = [abs(r['delta']['p_ap']) for r in rows]
    summary = dict(
        checkpoint=os.path.basename(os.path.dirname(CKPT)),
        arm='hic seed1 realiad', n_cats=N_CATS, n_imgs_per_cat=N_IMGS,
        abs_delta_f1max_max=max(d_f1), abs_delta_p_auroc_max=max(d_au),
        abs_delta_p_ap_max=max(d_ap),
        rows=rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(summary, open(OUT, 'w', encoding='utf-8'), indent=2)
    print(f"[audit] saved {OUT}")
    print(f"[audit] max |dF1|={max(d_f1):.5f}  max |dAUROC|={max(d_au):.5f}  "
          f"max |dAP|={max(d_ap):.5f}")


if __name__ == '__main__':
    main()
