# SGHL — Signal-Gated Halo Learning for Cross-Dataset Defect Segmentation

Official implementation of **"Signal-Gated Halo Learning: Boundary-Geometry
Supervision against Segmentation-Head Collapse in Cross-Dataset Defect
Segmentation"** (Hu, 2026; under review).

## What is this?

Standard Dice/BCE training of a segmentation head **collapses to a
spatially constant trivial solution** under cross-dataset supervision
with extreme positive-pixel scarcity (defects cover 0.1–0.3% of pixels):
across MVTec-AD, VisA, and Real-IAD, baselines collapse on **8 of 9**
dataset×seed slots, and neither pos-weight reweighting nor a best-effort
focal loss prevents it.

**HIC (Halo-Interior Contrastive)** supervises the *boundary geometry* of
the energy map instead:

```
L_HIC = [ p_B − p_I + m ]₊  −  log( p_I / (p_I + p_H) )
```

with interior I = erode₃(G), boundary B = G∖I, halo H = dilate₅(G)∖G
(region means, margin m = 0.3, min-area guard 32 px). On a frozen
DINOv2-reg ViT-B/14 + 2.75M UNet head, HIC **eliminates collapse on
12 of 12 slots** and leads the supervised-transfer family
(+0.112 / +0.068 P-F1 over MultiADS on MVTec / VisA). A dosing
principle governs the suppressive term (halo suppression: on for
synthetic supervision, exactly zero for real GT).

| Target | Arm | P-F1 | I-AUROC | AUPRO | Collapsed |
|---|---|---|---|---|---|
| MVTec | base_pw | 0.111 | 0.500 | 0.000 | 3/3 |
| MVTec | **hic** | **0.352** | **0.815** | 0.219 | **0/3** |
| VisA | base_pw | 0.035 | 0.500 | 0.000 | 3/3 |
| VisA | **hic** | **0.291** | **0.773** | 0.247 | **0/3** |
| Real-IAD | base | 0.169±0.26 | 0.647 | 0.441 | 2/3 |
| Real-IAD | **hic_adp** | **0.258** | **0.777** | 0.310 | **0/3** |

(Full metrics in `lineC_sup/results/*.json`; multi-source LODO,
seeds 1–3, mean±range.)

## Repository layout

```
lineC_sup/            Main line: LODO cross-dataset supervised segmentation
  train.py            Training (frozen DINOv2 + UNet head; arms: base,
                      base_pw, focal, hic, sghl, hic_adp, hic_m01/...)
  eval.py             Frozen evaluation protocol (exact P-F1max, AUPRO,
                      class bootstrap CIs)
  losses.py           HIC / HS / focal losses + adaptive morphology
  models.py datasets.py
  run_linec_pipeline6.py   Example watchdog queue (focal arm study)
  run_linec_pipeline7.py   Example watchdog queue (A5 sensitivity OFAT)
  results/            All frozen result JSONs (61 runs)
lineB_weak/           Weak-supervision line (k-shot fine-tuning study)
results/              Mechanism-analysis data (λ shrinkage, region
                      energies, gating prior) backing the paper figures
figures/              Figure-generation scripts + generated figures
eval/EVAL_PROTOCOL.md The frozen evaluation protocol (v1)
```

## Setup

```bash
pip install torch torchvision numpy scipy scikit-learn pillow
# DINOv2-reg ViT-B/14 weights download automatically via torch.hub
```

Tested: Python 3.11, PyTorch 2.x, single RTX 3090 (24 GB).

## Data preparation (required — not redistributed here)

The three benchmarks are **not** included; download from the official
sources and agree to their licenses:

| Dataset | Source | Please cite |
|---|---|---|
| MVTec-AD | https://www.mvtec.com/company/research/datasets/mvtec-ad | Bergmann et al., CVPR 2019 |
| VisA | https://registry.opendata.aws/visa/ | Zou et al., ECCV 2022 |
| Real-IAD | https://realiad.adec-technologies.com/ | Wang et al., CVPR 2024 |

Expected layout (edit `datasets.py` paths for your machine):

```
DATA_ROOT/
  mvtec/{bottle,...}/{train,test}/...
  visa/{candle,...}/...
  realiad/          # run lineC_sup/resize_realiad_448.py to build the
                    # pre-scaled 448px copy used by the protocol
```

## Reproducing the main results

```bash
cd lineC_sup
# one arm, one direction, one seed (~80 min on a 3090):
python train.py --train_sets visa realiad --test_set mvtec --arm hic \
    --epochs 5 --bs 12 --lr 1e-4 --seed 1
python eval.py  --train_sets visa realiad --test_set mvtec --arm hic --seed 1

# full queue (focal study): python run_linec_pipeline6.py
# sensitivity OFAT:          python run_linec_pipeline7.py
```

Every number in the paper is regenerated deterministically from the
JSONs in `lineC_sup/results/` under the protocol in
`eval/EVAL_PROTOCOL.md`.

## Notes

- `lineC_sup/BUGFIXES.md` documents two evaluation pitfalls found and
  fixed during development (source-domain anomaly pooling; AUPRO
  implementation) — read before re-running baselines.
- Figure scripts in `figures/` expect the analysis workspaces
  (`results/lambda_shrinkage`, `results/overfire_check`, and energy-map
  samples that are too large to ship); they are included for
  documentation of figure provenance.

## License

Code: MIT (see `LICENSE`). Datasets remain under their own licenses and
must be obtained from the official sources above.

## Citation

```bibtex
@misc{hu2026sghl,
  title  = {Signal-Gated Halo Learning: Boundary-Geometry Supervision
            against Segmentation-Head Collapse in Cross-Dataset Defect
            Segmentation},
  author = {Hu, Xiaojun},
  year   = {2026},
  note   = {Under review}
}
```
