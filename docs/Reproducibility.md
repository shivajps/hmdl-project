# Reproducibility Guide

This document maps each manuscript claim to the code/config that produces it.

| Manuscript item | Script | Config |
|---|---|---|
| Section 5.1 preprocessing, Table 2a | `preprocessing/preprocessing.py` | `configs/experiment_settings.yaml` |
| Section 4.3 CLF-GWO, Section 4.3.4 settings | `feature_selection/clf_gwo.py` | `configs/hyperparameters.yaml` |
| Section 4.4–4.6 model architecture | `models/iot_shield.py` | `configs/model_config.yaml` |
| Section 5.3 training (10 runs, 5 seeds) | `training/train.py` | `configs/model_config.yaml` |
| Table 3 (Accuracy/Precision/Recall/F1/MCC/Kappa) | `training/evaluate.py` | — |
| Section 6.3 / Table 4 latency benchmarking | `training/inference.py` | — |
| Table 5 (adversarial robustness) | *(add adversarial eval script here once implemented)* | — |
| Table 6/6a/6b (ablation) | *(add ablation runner script here once implemented)* | — |
| Section 6.5 (zero-day evaluation) | *(add open-set eval script here once implemented)* | — |
| Section 8.1 (SHAP/attention/GCN fidelity) | *(add explainability eval script here once implemented)* | — |

## Environment

\`\`\`bash
conda env create -f environment.yml
conda activate iot-shield
\`\`\`

## Hardware tiers referenced in the paper (Section 5.4/6.3)

| Tier | Description |
|---|---|
| H1 | Server (e.g., A100/similar GPU) |
| H2 | Jetson-class edge device |
| H3 | Raspberry Pi 4B, INT8-quantized |

## Notes for reviewers

Scripts marked *(add ... once implemented)* above are placeholders for
evaluations described in the current manuscript revision. If you are reading
this as a reviewer and a script is still a placeholder, treat the
corresponding manuscript table as **pending verification** rather than final.