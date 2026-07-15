"""
Z-score normalization fit only on the training partition (see manuscript
Section 5.1 — leakage-prevention procedure).
"""
import pandas as pd


def fit_normalizer(train_df: pd.DataFrame, exclude_cols=None):
    exclude_cols = exclude_cols or []
    numeric_cols = [c for c in train_df.columns
                    if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    means = train_df[numeric_cols].mean()
    stds = train_df[numeric_cols].std().replace(0, 1.0)
    return {"cols": numeric_cols, "means": means, "stds": stds}


def apply_normalizer(df: pd.DataFrame, normalizer: dict, exclude_cols=None) -> pd.DataFrame:
    df = df.copy()
    cols, means, stds = normalizer["cols"], normalizer["means"], normalizer["stds"]
    df[cols] = (df[cols] - means) / stds
    return df