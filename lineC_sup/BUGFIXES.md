# Line C 管线 Bug 修复记录 (2026-08-15)

排查范围: `lineC_sup/` 三向 × 四臂首轮结果异常。修复后由看门狗 3
(`run_linec_pipeline3.py`, 2026-08-15 11:00 启动) 全量重跑 12 作业。

## Bug 1: MVTec/VisA 训练池漏掉全部异常标注（训练侧，严重)

**根因**:`datasets.py` 的 `MVTecStyleAdapter.train_samples()` 只返回 `train/good`
正常图。MVTec/VisA 这类 AD 数据集的像素级异常标注在 **test split**
(`test/{defect}/` + `ground_truth/`),train split 设计上只有正常图。后果:

- D3 {MVTec+VisA}→Real-IAD 训练池 = 12288 张**全正常图，零异常** → 四臂全部
  未见过缺陷 (P-F1≈0.005, P-AUC 0.33~0.54);
- D1/D2 训练池缺 VisA/MVTec 的 1200/1258 张源域异常 (只有 Real-IAD 异常在场)。

**排除项（有可视化/数值证据）**：掩码-图像**无错位**（5 张叠加图逐一肉眼确认，
掩码精确落在缺陷上，见 `diag/realiad_overlay_*.png`)；标签**无反转**（正常样本
mask_path 全为 None、异常样本全有掩码；Raw 掩码为调色板/灰度非二值 PNG，
`(t>0)` 二值化正确）；多相机摊平正确（官方 JSON 路径与磁盘 3865/3865 一致）。

**修复**:`train_samples()` 追加 test split 带掩码异常图（LODO 无泄漏：评测数据集
整体不进训练池）。修正后池：D1=97653(VisA 9859 含 1200 异常）、D2=92681
(MVTec 4887 含 1258 异常）、D3=14746（含 2458 异常）。

**证据**：金丝雀 D3 hic 全量重训 (12.8min) 后 Real-IAD 前两类：
audiojack F1 0.002→0.113 / AUPRO 0.067→0.186;bottle_cap F1→0.104 / AUPRO→0.103;
预测热力图确认模型在真实缺陷处点火（`diag/realiad_pred_*.png`)。残留弱点
（正常结构圆环误报、P-AUC<0.7）属 5-epoch 快速日程的模型强度问题，非数据管线 bug。

## Bug 2: AUPRO 口径错误 + 直方图 AP 符号反转（评估侧）

**根因 A (AUPRO)**：首版 `compute_aupro` 与 MVTec 官方实现有两处偏差：
1. 积分归一化用固定 `/0.3`，而官方 (`CostFilter utils.compute_pro_original`) 是
   滤 `fpr<0.3` 后 **`fpr /= fpr.max()` 重标定到 [0,1]** 再求 auc——对双峰分数
   分布（背景~0、缺陷~1)，线性阈值在 FPR≤0.3 区间覆盖极窄，面积被压成 ~0.01;
2. FPR 负像素只算了异常图，官方含**正常图全部像素**；阈值也应取全测试图
   min/max 的 `np.arange`。

**数值自验**：修复后向量化实现 vs 官方循环忠实复刻，bottle/cable 两类
diff=0.00e+00;bottle hic AUPRO 0.0127→**0.2947**，与手工抽查 PRO(t\*)=0.58@
FPR=3% 自洽。向量化 0.4s/类 vs 官方循环 27.6s/类。

**根因 B (AP 符号）**:`binned_pixel_metrics_from_hist` 的 AP 用
`rec − rec[i−1]`（沿 recall 递减方向），符号反转；大类（Real-IAD 方向，bins4096
路径）出现负 AP（金丝雀实测 −0.04)。修为 `rec − rec[i+1]`。合成数据四档分离度
对照 sklearn `average_precision_score`,diff ≤ 0.0006。

**影响面**:AUPRO 列影响全部 12 个 JSON;AP 符号仅影响 Real-IAD 方向
（唯一走 bins4096 路径的评测集）。F1max/AUROC 路径无误。

## 重跑清单（看门狗 3,PID 启动时记录于 pipeline_status.txt)

| 方向 | 作业 | 说明 |
|---|---|---|
| D3 {M+V}→Real-IAD | hic（仅评估， 金丝雀 ckpt)/base/sghl/base_pw（重训+评估） | 4 |
| D1 {V+R}→MVTec | 四臂全量重训+评估 | 4 |
| D2 {M+R}→VisA | 四臂全量重训+评估 | 4 |

**存档**：旧 JSON → `results/*_buggy.json`（12 个）；旧 ckpt →
`checkpoints/*_wrongpool/`(12 个）。新结果覆盖原名（协议命名不变）。

## 备注

- eval.py 保留了 2026-08-14 23:31 另一会话加入的 NaN 防护（塌缩头输出 NaN → 0)。
- D3 方向金丝雀 P-AUROC 未达 0.7(audiojack 0.45 / bottle_cap 0.31)：根因是
  5-epoch 快速日程 + Real-IAD 缺陷极小（r_pos≈0.0005-0.001）与误报抑制不足，
  属训练强度问题；若主表需要，后续加长日程或调 lr,非管线 bug。
