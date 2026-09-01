#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R1 3.2 审稿意见: 自适应形态学硬阈值 A>=1568px 附近的样本占比统计.

A 的语义与训练完全一致 (losses.hic_adp_loss_per_sample):
  mask -> resize 448 (NEAREST) -> center crop 392 -> (t>0) -> A = 非零像素数.
统计每个数据集全部异常 mask 的 A 分布, 输出阈值带内占比.
输出: results/adp_threshold_stats.json
"""
import json
import os

import numpy as np
from PIL import Image

from datasets import (CROP_SIZE, IMAGE_SIZE, get_adapter)

TH = 1568
OUT = 'results/adp_threshold_stats.json'


def area_of(path):
    m = Image.open(path).convert('L')
    m = m.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    w = (IMAGE_SIZE - CROP_SIZE) // 2
    m = m.crop((w, w, w + CROP_SIZE, w + CROP_SIZE))
    return int((np.asarray(m) > 0).sum())


def stats_for(areas):
    a = np.asarray(areas, dtype=float)
    return dict(
        n=len(a),
        median=float(np.median(a)),
        p5=float(np.percentile(a, 5)),
        p95=float(np.percentile(a, 95)),
        frac_below_th=round(float((a < TH).mean()), 4),
        frac_band_pm25=round(float(((a >= TH * 0.75) & (a <= TH * 1.25)).mean()), 4),
        frac_band_pm12=round(float(((a >= TH * 0.88) & (a <= TH * 1.12)).mean()), 4),
    )


def main():
    res = {}
    for ds in ('mvtec', 'visa'):
        ad = get_adapter(ds)
        samples = [s for s in ad.test_samples() if s['label'] == 1 and s['mask_path']]
        areas = [area_of(s['mask_path']) for s in samples]
        res[ds] = stats_for(areas)
        print(ds, res[ds], flush=True)

    ad = get_adapter('realiad')
    samples = [s for s in ad.test_samples() if s['label'] == 1 and s['mask_path']]
    rng = np.random.default_rng(0)
    if len(samples) > 4000:
        idx = rng.choice(len(samples), 4000, replace=False)
        samples = [samples[i] for i in idx]
    areas = [area_of(s['mask_path']) for s in samples]
    res['realiad'] = stats_for(areas)
    res['realiad']['note'] = f'random 4000 of full anomalous set'
    print('realiad', res['realiad'], flush=True)

    os.makedirs('results', exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print('saved', OUT)


if __name__ == '__main__':
    main()
