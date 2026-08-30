#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Line C 损失: base = Dice+BCE; sghl = Dice+BCE + HIC + HS.

HIC/HS 形态学与 halo_losses_costfilter.py (论文 Section 10) 一致:
    halo 带  H = dilate_5(M) \\ M
    interior I = erode_3(M)
    boundary B = M \\ I
max_pool 近似形态学; HIC margin=0.3.

与 CostFilter 版差异: 逐样本计算 + min-area guard —— interior 像素 <32 的样本
其 HIC/HS 权重置 0 (设计文档第五节风险表), 然后对有效样本取均值.
prob = sigmoid(logits) (单通道分割头, 直接 sigmoid 即"预测能量").
"""
import torch
import torch.nn.functional as F

MIN_INTERIOR_AREA = 32
HIC_MARGIN = 0.3
# base_pw 臂: BCE 正样本权重 (不平衡修正). 取 10 的依据: 池内 batch 级正像素率
# 实测 ~0.1-0.3%, 理论全平衡权重 ~300-1000 会把梯度主导权完全交给缺陷像素
# (误报爆炸); 10 是"让正样本梯度进入有效区间但不压过背景"的常用温和档,
# 与设计文档 C-base 定义 (Dice+BCE) 偏差最小.
POS_WEIGHT = 10.0

# focal 臂 (2026-08-24, REVIEW_V1 A2 对照实验): Dice + Binary Focal, 回答
# "focal 式重加权是否同样无法对抗塌缩" (论文 Sec 2.2 断言的实验支撑).
# 配置选择 (best-effort, 防"故意削弱 focal"质疑): gamma=2 (Lin et al. ICCV'17
# 标准值); alpha=0.75 给正类 3x 权重 —— 让 focal 同时具备 base_pw 的正向强调
# 与聚焦项, 若仍塌缩则"幅度修正 ≠ 几何修正"的论证在其最强配置下成立.
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75


def focal_loss(logits, gt, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    """Binary focal loss: -alpha_t * (1-p_t)^gamma * log(p_t).

    alpha_t = alpha (正类) / 1-alpha (负类); p_t = p (y=1) / 1-p (y=0).
    """
    p = torch.sigmoid(logits)
    pos = (gt > 0.5).float()
    p_t = pos * p + (1.0 - pos) * (1.0 - p)
    alpha_t = pos * alpha + (1.0 - pos) * (1.0 - alpha)
    fl = -alpha_t * (1.0 - p_t).pow(gamma) * torch.log(p_t.clamp(min=1e-8))
    return fl.mean()


def make_region_masks(gt, halo_radius=5, interior_erode=3):
    """gt (B,1,H,W) 0/1 -> (halo, interior, boundary), max_pool 形态学近似."""
    gt_b = (gt > 0.5).float()
    k = 2 * halo_radius + 1
    dilated = F.max_pool2d(gt_b, kernel_size=k, stride=1, padding=halo_radius)
    halo = (dilated - gt_b).clamp(min=0.0)
    ke = 2 * interior_erode + 1
    eroded = 1.0 - F.max_pool2d(1.0 - gt_b, kernel_size=ke, stride=1, padding=interior_erode)
    interior = (eroded * gt_b).clamp(min=0.0)
    boundary = (gt_b - interior).clamp(min=0.0)
    return halo, interior, boundary


def dice_loss(logits, gt, eps=1e-6):
    prob = torch.sigmoid(logits)
    num = 2 * (prob * gt).sum(dim=(2, 3)) + eps
    den = prob.sum(dim=(2, 3)) + gt.sum(dim=(2, 3)) + eps
    return (1 - num / den).mean()


def bce_loss(logits, gt):
    return F.binary_cross_entropy_with_logits(logits, gt)


def hic_loss_per_sample(prob, gt, margin=HIC_MARGIN, halo_radius=5,
                        interior_erode=3):
    """逐样本 HIC = relu(mean(p[B]) - mean(p[I]) + margin)
                  - log( mean(p[I]) / (mean(p[I]) + mean(p[H])) )
    返回 (per_sample_loss (B,), valid (B,) — interior>=MIN_INTERIOR_AREA 且 halo 非空)."""
    halo, interior, boundary = make_region_masks(gt, halo_radius, interior_erode)
    area_i = interior.sum(dim=(2, 3))
    area_h = halo.sum(dim=(2, 3))
    area_b = boundary.sum(dim=(2, 3))
    p_i = (prob * interior).sum(dim=(2, 3)) / area_i.clamp(min=1.0)
    p_h = (prob * halo).sum(dim=(2, 3)) / area_h.clamp(min=1.0)
    p_b = (prob * boundary).sum(dim=(2, 3)) / area_b.clamp(min=1.0)
    eps = 1e-6
    loss = F.relu(p_b - p_i + margin) - torch.log(p_i / (p_i + p_h + eps) + eps)
    valid = ((area_i >= MIN_INTERIOR_AREA) & (area_h > 0)).float().squeeze(1)
    return loss.squeeze(1), valid


# ---- A5 敏感性扫描变体 (2026-08-24, REVIEW_V1 A5): 伪臂名承载超参,
# tag/JSON 命名天然区分, 其余与 hic 完全一致. OFAT 自基线 (m=0.3, r=5). ----
HIC_VARIANTS = {
    'hic_m01': dict(margin=0.1),           # margin 下探
    'hic_m05': dict(margin=0.5),           # margin 上探
    'hic_r3':  dict(halo_radius=3),        # halo 环带收窄
    'hic_r8':  dict(halo_radius=8),        # halo 环带加宽
}


def hs_loss_per_sample(prob, gt):
    """逐样本 HS = mean(prob[halo带]); valid 同 HIC guard (interior>=32 且 halo 非空)."""
    halo, interior, _ = make_region_masks(gt)
    area_h = halo.sum(dim=(2, 3))
    area_i = interior.sum(dim=(2, 3))
    loss = ((prob * halo).sum(dim=(2, 3)) / area_h.clamp(min=1.0)).squeeze(1)
    valid = ((area_i >= MIN_INTERIOR_AREA) & (area_h > 0)).float().squeeze(1)
    return loss, valid


def _masked_mean(per_sample, valid):
    denom = valid.sum().clamp(min=1.0)
    return (per_sample * valid).sum() / denom


# ---- hic_adp: 自适应形态学 (2026-08-16, Real-IAD 小缺陷修复) ----
# 机理背景: {M+V}->Real-IAD 方向 hic 30/30 类受损. Real-IAD 缺陷极小
# (r_pos≈0.001, 392^2 下常只有几十~几百像素): 固定 erode=3 把 interior 腐蚀空
# (触发 min-area guard, HIC 失效); 固定 halo=5 的带尺度≈缺陷本体, "压外圈"变成
# 压缺陷本体. 自适应规则 (按每样本 GT 面积 A, 392^2 像素计):
#   erode: A >= 32*(2*3+1)^2 = 1568 时用 3, 否则退到 1 (保 interior 存活)
#   halo_radius: min(5, max(2, int(sqrt(A)/8)))  (带宽随缺陷尺度收缩)
#   guard: A < 32 时该样本 HIC 置零 (沿用 min-area guard 语义);
#          另保留数值保护 interior==0 或 halo==0 时置零 (防 -log(0) 爆值).
ADP_ERODE = 3
ADP_HALO = 5
ADP_ERODE_MIN_AREA = MIN_INTERIOR_AREA * (2 * ADP_ERODE + 1) ** 2  # 1568


def _hic_terms(prob, gt, halo_radius, erode, margin):
    """给定形态学参数, 返回逐样本 (hic_loss, interior_area, halo_area)."""
    halo, interior, boundary = make_region_masks(gt, halo_radius, erode)
    area_i = interior.sum(dim=(2, 3))
    area_h = halo.sum(dim=(2, 3))
    area_b = boundary.sum(dim=(2, 3))
    p_i = (prob * interior).sum(dim=(2, 3)) / area_i.clamp(min=1.0)
    p_h = (prob * halo).sum(dim=(2, 3)) / area_h.clamp(min=1.0)
    p_b = (prob * boundary).sum(dim=(2, 3)) / area_b.clamp(min=1.0)
    eps = 1e-6
    loss = F.relu(p_b - p_i + margin) - torch.log(p_i / (p_i + p_h + eps) + eps)
    return loss.squeeze(1), area_i.squeeze(1), area_h.squeeze(1)


def hic_adp_loss_per_sample(prob, gt, margin=HIC_MARGIN):
    """自适应形态学 HIC, 逐样本. 返回 (per_sample_loss (B,), valid (B,))."""
    area = gt.sum(dim=(2, 3)).squeeze(1)                       # A
    erode = torch.where(area >= ADP_ERODE_MIN_AREA,
                        torch.full_like(area, ADP_ERODE),
                        torch.ones_like(area))
    halo_r = (area.sqrt() / 8).int().clamp(min=2, max=ADP_HALO)
    # batch 内按 (erode, halo) 分组共享 max_pool kernel (erode∈{1,3}, halo∈{2..5}),
    # 分组结果用可微 scatter 聚合回 (B,)
    poss, vals, valids = [], [], []
    for e in (1, ADP_ERODE):
        for h in range(2, ADP_HALO + 1):
            idx = ((erode == e) & (halo_r == h)
                   & (area >= MIN_INTERIOR_AREA)).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            l, ai, ah = _hic_terms(prob[idx], gt[idx], h, e, margin)
            poss.append(idx)
            vals.append(l)
            valids.append(((ai > 0) & (ah > 0)).float())
    if not vals:
        return torch.zeros_like(area), torch.zeros_like(area)
    pos = torch.cat(poss)
    out = torch.zeros_like(area).scatter(0, pos, torch.cat(vals))
    valid = torch.zeros_like(area).scatter(0, pos, torch.cat(valids))
    return out, valid


def compute_losses(logits, gt, arm='base', lambda_hs=1.0, lambda_hic=1.0):
    """返回 dict(total, dice, bce, hic, hs, n_valid_hic, n_valid_hs).

    arm='base': Dice+BCE; arm='hic': +HIC (无 HS); arm='sghl': + HIC + HS
    (HS 常数权重 1.0, 门控后加); arm='base_pw': Dice+BCE(pos_weight=POS_WEIGHT).
    min-area guard: interior<32px 样本 HIC/HS 权重置 0 (在 *_per_sample 内实现).

    base_pw 实现选择说明 (pos_weight vs Focal+SSIM):
      CostFilter 原基线用 FocalLoss+SSIM, 但主消融是 base_pw vs hic —— 要求对照臂
      与 base/hic 的**损失族严格一致 (Dice+BCE)**, 唯一变量是不平衡修正. Focal+SSIM
      会同时改变损失形式 (gamma 聚焦 + 结构相似项), 引入两个新超参, 使
      "base_pw vs hic" 的差异不再能归因于 HIC. pos_weight 是 BCEWithLogits 的
      原生单参数, 直接针对已诊断的塌缩机制 (正像素率 ~0.1-0.3%, 背景梯度淹没),
      改动最小、归因最干净. 故选 pos_weight=10.
    """
    prob = torch.sigmoid(logits)
    dl = dice_loss(logits, gt)
    if arm == 'base_pw':
        bl = F.binary_cross_entropy_with_logits(
            logits, gt, pos_weight=logits.new_tensor(POS_WEIGHT))
    elif arm == 'focal':
        # focal: Dice + Focal(γ=2, α=0.75), 无 BCE/HIC/HS —— 幅度类修正的
        # 第三种代表 (base=无修正, base_pw=线性重加权, focal=非线性聚焦重加权).
        # bce 字段复用为 focal 值, train.py 日志列无需改动.
        bl = focal_loss(logits, gt)
    else:
        bl = bce_loss(logits, gt)
    out = dict(dice=dl, bce=bl)
    total = dl + bl
    if arm == 'sghl':
        hic_ps, v_hic = hic_loss_per_sample(prob, gt)
        hs_ps, v_hs = hs_loss_per_sample(prob, gt)
        hic = _masked_mean(hic_ps, v_hic)
        hs = _masked_mean(hs_ps, v_hs)
        out.update(hic=hic, hs=hs,
                   n_valid_hic=v_hic.sum().detach(), n_valid_hs=v_hs.sum().detach())
        total = total + lambda_hic * hic + lambda_hs * hs
    elif arm == 'hic' or arm in HIC_VARIANTS:
        # hic_only: Dice+BCE+HIC, 无 HS (Line B 剂量-损伤实验结论: 真实 GT 下 HS 伤 I-AUROC)
        # A5 变体 (hic_m01/m05/r3/r8): 仅改 margin/halo_radius, 见 HIC_VARIANTS
        hic_ps, v_hic = hic_loss_per_sample(prob, gt, **HIC_VARIANTS.get(arm, {}))
        hic = _masked_mean(hic_ps, v_hic)
        out.update(hic=hic, hs=torch.zeros((), device=logits.device),
                   n_valid_hic=v_hic.sum().detach(),
                   n_valid_hs=torch.zeros((), device=logits.device))
        total = total + lambda_hic * hic
    elif arm == 'hic_adp':
        # hic_adp: Dice+BCE+自适应形态学 HIC (规则见 hic_adp_loss_per_sample 注释;
        # Real-IAD 小缺陷修复: erode 3->1, halo 5->min(5,max(2,sqrt(A)/8)))
        hic_ps, v_hic = hic_adp_loss_per_sample(prob, gt)
        hic = _masked_mean(hic_ps, v_hic)
        out.update(hic=hic, hs=torch.zeros((), device=logits.device),
                   n_valid_hic=v_hic.sum().detach(),
                   n_valid_hs=torch.zeros((), device=logits.device))
        total = total + lambda_hic * hic
    else:
        z = torch.zeros((), device=logits.device)
        out.update(hic=z, hs=z, n_valid_hic=z, n_valid_hs=z)
    out['total'] = total
    return out
