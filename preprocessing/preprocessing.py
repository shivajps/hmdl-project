"""
Top-level preprocessing entrypoint for IoT-SHIELD.
Loads a raw dataset, applies cleaning, feature engineering, normalization,
class-imbalance handling, and produces train/val/test splits.

Usage:
    python preprocessing/preprocessing.py --dataset CICIDS2017 \
        --input raw/CICIDS2017/ --output processed/CICIDS2017/
"""
import argparse
import os
import pandas as pd

from feature_engineering import engineer_features
from normalization import fit_normalizer, apply_normalizer
from smote_enn import apply_smote_enn
from train_val_test_split import stratified_split


def load_raw(dataset_name: str, input_dir: str) -> pd.DataFrame:
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    frames = [pd.read_csv(os.path.join(input_dir, f), low_memory=False) for f in csv_files]
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                         choices=["CICIDS2017", "UNSW-NB15", "N-BaIoT", "TON_IoT", "Edge-IoT-2024"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--device-col", default=None,
                         help="Column used for device-level split stratification (Edge-IoT-2024 only)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = load_raw(args.dataset, args.input)
    df = engineer_features(df)

    train_df, val_df, test_df = stratified_split(
        df, label_col=args.label_col, device_col=args.device_col,
        train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=42,
    )

    normalizer = fit_normalizer(train_df, exclude_cols=[args.label_col])
    train_df = apply_normalizer(train_df, normalizer, exclude_cols=[args.label_col])
    val_df = apply_normalizer(val_df, normalizer, exclude_cols=[args.label_col])
    test_df = apply_normalizer(test_df, normalizer, exclude_cols=[args.label_col])

    train_df = apply_smote_enn(train_df, label_col=args.label_col, seed=42)

    train_df.to_csv(os.path.join(args.output, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.output, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.output, "test.csv"), index=False)
    print(f"Saved processed splits to {args.output}")


if __name__ == "__main__":
    main()