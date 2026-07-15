"""
Feature engineering utilities: derived flow statistics, categorical encoding,
and missing-value handling. Adjust FEATURE_COLUMNS per dataset schema.
"""
import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how="any")

    # Example derived features — adapt to your actual flow schema.
    if {"total_fwd_bytes", "total_bwd_bytes"}.issubset(df.columns):
        df["byte_ratio"] = df["total_fwd_bytes"] / (df["total_bwd_bytes"] + 1e-6)

    if {"flow_duration", "total_fwd_packets"}.issubset(df.columns):
        df["packets_per_sec"] = df["total_fwd_packets"] / (df["flow_duration"] + 1e-6)

    # Encode categorical columns (protocol, service, state, etc.)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    label_col_candidates = {"label", "attack_cat", "type"}
    categorical_cols = [c for c in categorical_cols if c.lower() not in label_col_candidates]

    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes

    return df