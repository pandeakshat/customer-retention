"""
load.py — Dataset loading, auditing, schema confirmation (generic, not Telco-only)

This module centralizes dataset I/O and light cleaning. It supports arbitrary
CSV schemas by providing helpers to normalize columns and (optionally) coerce
a known Telco-style "TotalCharges" column. Target/ARPU/tenure selection is
kept dynamic in the app so the module remains dataset-agnostic.

Author: Customer Intelligence Hub
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_PATH = os.path.join("data", "sample.csv")


# ---------- Public API ----------

def load_dataset(path: str = DEFAULT_PATH, uploaded_file: Optional[object] = None) -> pd.DataFrame:
    """
    Load a CSV dataset either from a provided Streamlit upload or from disk.

    Parameters
    ----------
    path : str
        Fallback CSV path (relative to repo root).
    uploaded_file : object | None
        File-like object from Streamlit file_uploader.

    Returns
    -------
    pd.DataFrame
    """
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'. Upload a CSV or place a sample at that path.")
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize columns to snake_case-like (spaces -> underscores). No typing changes.
    """
    out = df.copy()
    out.columns = [c.strip().replace(" ", "_") for c in out.columns]
    return out


def coerce_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort coercion of TotalCharges-like columns (common in Telco CSVs).
    Safe no-op for datasets without these columns.
    """
    out = df.copy()
    for cand in ["TotalCharges", "Total_Charges", "total_charges"]:
        if cand in out.columns:
            out[cand] = pd.to_numeric(out[cand].replace(" ", np.nan), errors="coerce")
    return out


def minimal_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply light, lossless cleaning suitable for arbitrary churn datasets:
    - Standardize columns (underscores)
    - Best-effort numeric coercion for TotalCharges-like fields (safe no-op otherwise)
    """
    out = standardize_columns(df)
    out = coerce_total_charges(out)
    return out


def audit_dataset(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, object]:
    """
    Produce a lightweight audit dict for structure and quality indicators.

    Returns
    -------
    dict {
        "shape": (rows, cols),
        "dtypes": {col: dtype, ...},
        "missing_perc": {col: pct_missing, ...},
        "class_balance": {label: count, ...} or None
    }
    """
    out: Dict[str, object] = {}
    out["shape"] = df.shape
    out["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    out["missing_perc"] = df.isna().mean().sort_values(ascending=False).to_dict()

    if target_col and target_col in df.columns:
        out["class_balance"] = df[target_col].value_counts(dropna=False).to_dict()
    else:
        out["class_balance"] = None
    return out


def detect_binary_target_candidates(df: pd.DataFrame, candidate_names: Optional[List[str]] = None) -> List[str]:
    """
    Suggest columns that look like binary classification targets.

    Heuristics:
      - exactly 2 unique non-NaN values
      - column name hints (e.g., 'churn', 'cancel', 'active')

    Returns
    -------
    List[str]
    """
    hints = {"churn", "cancel", "attrit", "left", "exited", "active"}
    if candidate_names is None:
        candidate_names = [c for c in df.columns if any(h in c.lower() for h in hints)]
    binary_cols = []
    for c in df.columns:
        non_na = df[c].dropna()
        if non_na.nunique() == 2:
            binary_cols.append(c)
    # Prioritize hinted names
    prioritized = [c for c in candidate_names if c in df.columns and c in binary_cols]
    remaining = [c for c in binary_cols if c not in prioritized]
    return prioritized + remaining
