"""
SMOTE-ENN oversampling for class-imbalance handling.
Fit and applied ONLY on the training partition.
"""
from imblearn.combine import SMOTEENN
import pandas as pd


def apply_smote_enn(train_df: pd.DataFrame, label_col: str, seed: int = 42) -> pd.DataFrame:
    X = train_df.drop(columns=[label_col])
    y = train_df[label_col]

    sme = SMOTEENN(random_state=seed)
    X_res, y_res = sme.fit_resample(X, y)

    out = pd.DataFrame(X_res, columns=X.columns)
    out[label_col] = y_res
    return out