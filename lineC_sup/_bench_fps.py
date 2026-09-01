#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R1 3.1 审稿意见: RTX 3090 推理吞吐/延迟/显存基准.

级联 = 冻结 DINOv2-reg ViT-B/14 (86M) + UNet 分割头 (2.75M 可训练).
输入 392x392x3 (训练/评估分辨率). 精度: fp16 autocast (与 eval.py 部署一致)
及 fp32 对照. 输出 JSON: results/fps_bench.json
"""
import json
import time

import torch

from models import LineCModel

CKPT = 'checkpoints/mvtec+realiad_to_visa_hic_seed1/last.pth'
OUT = 'results/fps_bench.json'


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    model = LineCModel('cuda')
    sd = torch.load(CKPT, map_location='cpu')
    model.head.load_state_dict(sd['head'])
    model.eval()

    n_bb = sum(p.numel() for p in model.encoder.parameters())
    n_head = sum(p.numel() for p in model.head.parameters())
    res = {
        'gpu': torch.cuda.get_device_name(0),
        'input': '3x392x392 (train/eval resolution)',
        'params_backbone_frozen_M': round(n_bb / 1e6, 2),
        'params_head_trainable_M': round(n_head / 1e6, 3),
    }

    for bs in (1, 8):
        x = torch.randn(bs, 3, 392, 392, device='cuda')
        for amp in (True, False):
            def full():
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                    model(x)
            iters = 200 if bs == 1 else 100
            torch.cuda.reset_peak_memory_stats()
            dt = bench(full, 20, iters)
            peak = torch.cuda.max_memory_allocated() / 2**20
            res[f'b{bs}_{"fp16" if amp else "fp32"}'] = {
                'fps': round(iters * bs / dt, 1),
                'ms_per_img': round(dt / (iters * bs) * 1e3, 2),
                'vram_peak_MB': round(peak, 0),
            }

    # 拆解: backbone vs head (fp16, b=1)
    x = torch.randn(1, 3, 392, 392, device='cuda')
    def bb():
        with torch.autocast('cuda', dtype=torch.float16):
            model.extract(x)
    def head():
        with torch.autocast('cuda', dtype=torch.float16):
            f = model.extract(x)
            model.head(f)
    dt_bb = bench(bb, 20, 200)
    dt_full = bench(head, 20, 200)
    res['split_fp16_b1'] = {
        'backbone_ms': round(dt_bb / 200 * 1e3, 2),
        'full_ms': round(dt_full / 200 * 1e3, 2),
        'head_ms': round((dt_full - dt_bb) / 200 * 1e3, 2),
    }
    res['vram_reserved_MB'] = round(torch.cuda.memory_reserved() / 2**20, 0)

    import os
    os.makedirs('results', exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
