# IoT-SHIELD

**A Hybrid Metaheuristic Deep Learning Framework for Real-Time Intrusion Detection and Multi-Class Threat Classification in IoT Networks**

This repository contains the reference implementation accompanying the manuscript:

> [Author Names], "Hybrid Metaheuristic Deep Learning Approach for Real-Time Intrusion Detection and Threat Classification in Next-Generation IoT Networks," *[Journal Name]*, [Year], (under review/in press).

## Overview

IoT-SHIELD combines:
- **CLF-GWO** — a Chaotic Lévy-Flight Grey Wolf Optimizer for joint feature selection and hyperparameter tuning
- **TA-BiLSTM** — a temporal-attention bidirectional LSTM for sequential flow modelling
- **GCN** — a graph convolutional module for cross-device structural reasoning
- **Adaptive Fusion** — a learned fusion layer combining temporal and structural representations

## Repository Structure

See `docs/Reproducibility.md` for a full walkthrough of how each folder maps to the manuscript's sections and how to reproduce each reported table.

## Status

This repository is under active preparation alongside manuscript revision. Result artifacts under `results/` and `supplementary/` are populated as experiments are finalized — see each folder's README for current status.

## Installation

\`\`\`bash
git clone https://github.com/<your-username>/IoT-SHIELD.git
cd IoT-SHIELD
conda env create -f environment.yml   # or: pip install -r requirements.txt
\`\`\`

## Quick Start

\`\`\`bash
# 1. Preprocess a dataset
python preprocessing/preprocessing.py --dataset CICIDS2017 --config configs/experiment_settings.yaml

# 2. Run CLF-GWO feature selection
python feature_selection/clf_gwo.py --config configs/hyperparameters.yaml

# 3. Train IoT-SHIELD
python training/train.py --config configs/model_config.yaml

# 4. Evaluate
python training/evaluate.py --checkpoint <path_to_checkpoint> --dataset CICIDS2017
\`\`\`

## Datasets

This repository does **not** redistribute third-party datasets. See `datasets/README.md` and `datasets/dataset_links.txt` for official sources and access instructions for CICIDS2017, UNSW-NB15, N-BaIoT, TON_IoT, and Edge-IoT-2024.

## Citation

If you use this code, please cite the manuscript (see `CITATION.cff`).

## License

Released under the MIT License — see `LICENSE`.

## Contact

For questions regarding reproducibility, please open a GitHub issue or contact [author email].