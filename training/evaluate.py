"""
Evaluation entrypoint: computes Accuracy, Precision, Recall, F1, MCC, Kappa
(manuscript Table 3) on a held-out test set.
"""
import argparse
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix,
)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    raise NotImplementedError(
        "Load checkpoint + test set, run inference, then call compute_metrics(). "
        "Save output to results/tables/ as CSV — do not hand-fill these numbers."
    )


if __name__ == "__main__":
    main()