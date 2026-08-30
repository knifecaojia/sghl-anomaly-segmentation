# 评估协议 (冻结, v1, 2026-08-12)

> 本协议对 halo_ad 全部实验有约束力。任何主表数字必须按本协议产生。
> 依据: v5审稿建议.md 的教训 (下采样偏移 / F1max 近似误差 / 数值定义漂移)。

## 1. 指标计算

1. **P-F1max 必须精确**: 全像素排序或 ≥1000 档阈值扫描, 禁止 13 档 percentile 近似。
2. P-AUROC / P-AP / AUPRO / I-AUROC 按各数据集官方协议; AP 用 sklearn average_precision (全像素)。
3. AUPRO 按 MVTec 官方实现 (connected-component PRO, 积分限 FPR≤0.3)。

## 2. 评估总体

1. **固定测试集**: 每类使用官方完整测试集, 不做正常图下采样; 若必须子采样 (调试), 报告中
   明确标注 "subset", 且正式结果必须全量重跑。
2. 每类记录: n_normal / n_anomaly / 正像素比例 r_pos, 随结果 JSON 一并保存。

## 3. 统计严谨性

1. 跨类汇总报 macro 均值 + **bootstrap 95% CI** (以类为重采样单位, B=10000)。
2. 受控对比 (baseline vs halo) 必须: 同基座 checkpoint / 同训练日程 / 同评估脚本 / per-category
   配对差值表 (附录)。
3. 有随机性的环节 (coreset, 合成异常种子) 固定种子并记录; 关键结果 ≥2 种子复跑。

## 4. 结果存档

1. 每次评估输出 JSON 到 `results/`, 命名: `{date}_{line}_{base}_{dataset}_{config}_{seed}.json`。
2. JSON 必含: git/脚本版本、数据集与类别数、评估总体计数、全部指标、per-category 明细。
3. 日志同步存 `logs/`。

## 5. SOTA 对比表口径

1. 对比数字优先引自原论文; 无法对齐协议时在表注说明。
2. 主战场指标: P-F1max / P-AP / AUPRO; P-AUROC 必须报告 (证明不降)。
