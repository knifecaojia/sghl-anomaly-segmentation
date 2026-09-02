#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Line C 训练: frozen DINOv2-reg ViT-B/14 + 轻量 UNet 头, pooled 真实 GT.

用法:
    python train.py --train_sets visa realiad --test_set mvtec --arm sghl \
        --epochs 5 --bs 12 --lr 1e-4 --seed 1 [--smoke_steps 20]

- 数据集级平衡采样 (WeightedRandomSampler, 每样本权重 1/其数据集大小, 各数据集等概率);
- arm base: Dice+BCE; arm sghl: + HIC(margin=0.3) + HS(常数权重 1.0),
  min-area guard (interior<32px 权重置 0) 在 losses.py 内;
- 编码器冻结 + AMP fp16; 只训分割头; Adam lr=1e-4, grad clip 2.0;
- checkpoint 存 checkpoints/{run_tag}/epoch_{k}.pth 与 last.pth;
- 每 50 步打印 seg/hic/hs/total 运行均值; 每 epoch 打印汇总.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets as D
from losses import compute_losses
from models import LineCModel

LINE_C_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_tag(args):
    return f'{"+".join(args.train_sets)}_to_{args.test_set}_{args.arm}_seed{args.seed}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_sets', nargs='+', required=True,
                    choices=['mvtec', 'visa', 'realiad'])
    ap.add_argument('--test_set', required=True, choices=['mvtec', 'visa', 'realiad'])
    ap.add_argument('--arm', required=True,
                    choices=['base', 'sghl', 'hic', 'base_pw', 'hic_adp', 'focal',
                    'hic_m01', 'hic_m05', 'hic_r3', 'hic_r8',
                    'focal_a025', 'focal_a05'])
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--bs', type=int, default=12)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--smoke_steps', type=int, default=0,
                    help='>0 时只跑第 0 epoch 前 N 步, 不存 checkpoint')
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()

    tag = run_tag(args)
    ckpt_dir = os.path.join(LINE_C_DIR, 'checkpoints', tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    setup_seed(args.seed)
    print(f'[train] tag={tag} arm={args.arm} epochs={args.epochs} bs={args.bs} '
          f'lr={args.lr} seed={args.seed} smoke_steps={args.smoke_steps}', flush=True)

    t0 = time.time()
    pool = D.build_train_pool(args.train_sets)
    print(f'[train] pool size={len(pool)} '
          f'({ {d: sum(1 for s in pool if s["dataset_id"]==d) for d in args.train_sets} }, '
          f'anomaly={sum(s["label"] for s in pool)}) built in {time.time()-t0:.1f}s', flush=True)

    dataset = D.LineCDataset(pool)
    weights = D.dataset_balanced_weights(pool)
    sampler = WeightedRandomSampler(weights, num_samples=len(pool), replacement=True)
    loader = DataLoader(dataset, batch_size=args.bs, sampler=sampler,
                        num_workers=args.num_workers, collate_fn=D.collate_linec,
                        pin_memory=True, drop_last=True,
                        persistent_workers=args.num_workers > 0)

    model = LineCModel('cuda')
    opt = torch.optim.Adam(model.head.parameters(), lr=args.lr, betas=(0.9, 0.98))

    step = 0
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        run = dict(dice=0.0, bce=0.0, hic=0.0, hs=0.0, total=0.0, n=0)
        t_ep = time.time()
        for batch in loader:
            image = batch['image'].cuda(non_blocking=True)
            gt = batch['mask'].cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                feats = model.extract(image)
                logits = model.head(feats)
            out = compute_losses(logits.float(), gt, arm=args.arm)
            loss = out['total']
            if not torch.isfinite(loss):
                print(f'[train] SKIP non-finite loss at step {step}', flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 2.0)
            opt.step()

            for k in ('dice', 'bce', 'hic', 'hs', 'total'):
                run[k] += float(out[k])
            run['n'] += 1
            step += 1
            if step % 50 == 0:
                n = run['n']
                print(f'[train] ep{epoch} step{step} '
                      f'dice={run["dice"]/n:.4f} bce={run["bce"]/n:.4f} '
                      f'hic={run["hic"]/n:.4f} hs={run["hs"]/n:.4f} '
                      f'total={run["total"]/n:.4f}', flush=True)
            if args.smoke_steps and step >= args.smoke_steps:
                stop = True
                break
        n = max(run['n'], 1)
        print(f'[train] EPOCH {epoch} done in {(time.time()-t_ep)/60:.1f}min: '
              f'dice={run["dice"]/n:.4f} bce={run["bce"]/n:.4f} hic={run["hic"]/n:.4f} '
              f'hs={run["hs"]/n:.4f} total={run["total"]/n:.4f}', flush=True)
        if not args.smoke_steps:
            torch.save(dict(head=model.head.state_dict(), epoch=epoch, tag=tag,
                            args=vars(args)),
                       os.path.join(ckpt_dir, f'epoch_{epoch}.pth'))
            torch.save(dict(head=model.head.state_dict(), epoch=epoch, tag=tag,
                            args=vars(args)),
                       os.path.join(ckpt_dir, 'last.pth'))

    if args.smoke_steps:
        # 冒烟模式也存一个临时 checkpoint, 供冒烟评估使用 (不进正式 last.pth)
        torch.save(dict(head=model.head.state_dict(), epoch=-1, tag=tag,
                        args=vars(args)),
                   os.path.join(ckpt_dir, 'smoke.pth'))
        print(f'[train] smoke checkpoint saved: {ckpt_dir}\\smoke.pth', flush=True)

    print(f'[train] FINISHED tag={tag} steps={step} '
          f'total_time={(time.time()-t0)/60:.1f}min', flush=True)


if __name__ == '__main__':
    main()
