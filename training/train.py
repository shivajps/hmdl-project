"""
Training entrypoint for IoT-SHIELD.
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from random_seed import set_seed, SEEDS
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from iot_shield import IoTShield
from losses import FocalLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42, choices=SEEDS)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IoTShield(
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
        lstm_hidden=config.get("lstm_hidden", 128),
        lstm_layers=config.get("lstm_layers", 2),
        gcn_hidden=config.get("gcn_hidden", 64),
        gcn_layers=config.get("gcn_layers", 2),
        fused_dim=config.get("fused_dim", 128),
        dropout=config.get("dropout", 0.3),
    ).to(device)

    criterion = FocalLoss(gamma=config.get("focal_gamma", 2.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-3))

    raise NotImplementedError(
        "Wire in your DataLoader (train/val CSVs from preprocessing/) and "
        "training loop here — this entrypoint sets up model/optimizer/seed only."
    )


if __name__ == "__main__":
    main()