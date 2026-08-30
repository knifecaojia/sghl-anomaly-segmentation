#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Line C 模型: 冻结 DINOv2-reg ViT-B/14 特征 + 轻量 UNet 分割头.

- 骨干: torch.hub 本地缓存 (facebookresearch_dinov2_main) + 本地 reg4 权重,
  完全离线; 取中间层 [2,5,8,11] (B,768,28,28) @ 392x392 输入.
- 分割头: 每层 1x1 投影到 128 通道 -> concat(512) -> 3x3 融合 ->
  渐进上采样 28->56->112->224->392, 每级 upsample+conv, 输出 1 通道 logit.
  总参数量 ~1.2M (轻量).
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

LINE_C_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(LINE_C_DIR, 'weights', 'dinov2_vitb14_reg4_pretrain.pth')
FEATURE_LAYERS = [2, 5, 8, 11]
EMBED_DIM = 768


def load_frozen_dinov2(device='cuda'):
    hub_dir = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub',
                           'facebookresearch_dinov2_main')
    if not os.path.isdir(hub_dir):
        raise RuntimeError(f'torch hub dinov2 cache missing: {hub_dir}')
    model = torch.hub.load(hub_dir, 'dinov2_vitb14_reg', source='local', pretrained=False)
    sd = torch.load(WEIGHTS, map_location='cpu')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'backbone state dict mismatch: missing={missing}, unexpected={unexpected}')
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class UNetSegHead(nn.Module):
    """轻量 UNet 风格解码头: 多层冻结特征融合 -> 392x392 单通道 logit."""

    def __init__(self, n_levels=4, proj_dim=128, out_size=392):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(EMBED_DIM, proj_dim, 1) for _ in range(n_levels)])
        self.fuse = ConvBlock(proj_dim * n_levels, 256)          # 28x28
        self.up1 = ConvBlock(256, 128)                            # 56x56
        self.up2 = ConvBlock(128, 64)                             # 112x112
        self.up3 = ConvBlock(64, 32)                              # 224x224
        self.up4 = ConvBlock(32, 16)                              # 392x392
        self.out = nn.Conv2d(16, 1, 1)
        self.out_size = out_size

    def forward(self, feats):
        x = torch.cat([p(f) for p, f in zip(self.projs, feats)], dim=1)
        x = self.fuse(x)
        x = self.up1(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))
        x = self.up2(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))
        x = self.up3(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))
        x = self.up4(F.interpolate(x, size=(self.out_size, self.out_size),
                                   mode='bilinear', align_corners=False))
        return self.out(x)


class LineCModel(nn.Module):
    """冻结骨干 + 分割头; extract() 在 no_grad/autocast 下提特征."""

    def __init__(self, device='cuda'):
        super().__init__()
        self.encoder = load_frozen_dinov2(device)
        self.head = UNetSegHead().to(device)

    def extract(self, image):
        with torch.no_grad():
            feats = self.encoder.get_intermediate_layers(
                image, n=FEATURE_LAYERS, reshape=True)
        return feats

    def forward(self, image):
        return self.head(self.extract(image))
