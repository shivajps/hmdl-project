"""
Stratified 70/15/15 train/val/test split.
For Edge-IoT-2024, additionally stratifies by device_col to prevent the same
device's traffic from appearing in more than one partition (see manuscript
Section 5.1).
"""
from sklearn.model_selection import train_test_split
import pandas as pd


def stratified_split(df: pd.DataFrame, label_col: str, device_col=None,
                      train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    if device_col and device_col in df.columns:
        devices = df[device_col].unique()
        train_dev, temp_dev = train_test_split(devices, train_size=train_frac, random_state=seed)
        val_dev, test_dev = train_test_split(
            temp_dev, train_size=val_frac / (val_frac + test_frac), random_state=seed
        )
        train_df = df[df[device_col].isin(train_dev)]
        val_df = df[df[device_col].isin(val_dev)]
        test_df = df[df[device_col].isin(test_dev)]
    else:
        train_df, temp_df = train_test_split(
            df, train_size=train_frac, stratify=df[label_col], random_state=seed
        )
        val_df, test_df = train_test_split(
            temp_df, train_size=val_frac / (val_frac + test_frac),
            stratify=temp_df[label_col], random_state=seed,
        )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)