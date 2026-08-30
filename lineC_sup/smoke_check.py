#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Line C 冒烟检查: 三数据集适配器计数 + 骨干/头/损失形状验证."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import datasets as D
import losses as L
from models import LineCModel


def main():
    for did in ('mvtec', 'visa', 'realiad'):
        ad = D.get_adapter(did)
        cats = ad.categories()
        tr = ad.train_samples()
        te = ad.test_samples()
        n_an_tr = sum(s['label'] for s in tr)
        n_an_te = sum(s['label'] for s in te)
        print(f'[{did}] categories={len(cats)} train={len(tr)} (anomaly={n_an_tr}) '
              f'test={len(te)} (anomaly={n_an_te})')

    # 每数据集抽 1 个训练样本 + 1 个异常测试样本, 验证加载与形状
    samples = []
    for did in ('mvtec', 'visa', 'realiad'):
        ad = D.get_adapter(did)
        tr = ad.train_samples()
        te_an = [s for s in ad.test_samples() if s['label'] == 1]
        samples.append(tr[len(tr) // 2])
        samples.append(te_an[len(te_an) // 2])
    ds = D.LineCDataset(samples)
    batch = D.collate_linec([ds[i] for i in range(len(ds))])
    print('batch image:', tuple(batch['image'].shape), 'mask:', tuple(batch['mask'].shape),
          'mask px sums:', batch['mask'].sum(dim=(2, 3)).squeeze(1).tolist())
    print('labels:', batch['label'].tolist())
    print('ids:', list(zip(batch['dataset_id'], batch['category'])))

    model = LineCModel('cuda')
    n_head = sum(p.numel() for p in model.head.parameters())
    print(f'head params: {n_head/1e6:.2f}M')
    with torch.autocast('cuda', dtype=torch.float16):
        feats = model.extract(batch['image'].cuda())
        logits = model.head(feats)
    print('feats:', [tuple(f.shape) for f in feats], 'logits:', tuple(logits.shape))
    logits = logits.float()
    out = L.compute_losses(logits, batch['mask'].cuda(), arm='sghl')
    for k, v in out.items():
        print(f'  {k} = {v.item() if torch.is_tensor(v) and v.dim()==0 else v}')
    out['total'].backward()
    print('backward OK, no NaN:', all(torch.isfinite(v).all().item() for v in out.values()))
    w = D.dataset_balanced_weights(D.build_train_pool(['mvtec', 'visa']))
    print('balanced weights ok:', tuple(w.shape), float(w.sum()))


if __name__ == '__main__':
    main()
