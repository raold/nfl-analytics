"""
Off-policy evaluation estimators for one-step decisions.

Implements per-decision Self-Normalized Importance Sampling (SNIS) and a
lightweight Doubly Robust (DR) estimate with a simple outcome model.

Input DataFrame columns expected:
  action (0/1), r (float), b_prob (float in (0,1]), pi_prob (float in [0,1])
Optional feature columns for DR: any numeric covariates (e.g., spread_close,
total_close, epa_gap, market_prob, edge); non-numeric columns are ignored.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _clip(x: np.ndarray, c: float) -> np.ndarray:
    if c is None or c <= 0:
        return x
    return np.clip(x, 0.0, c)


def _shrink(w: np.ndarray, lam: float) -> np.ndarray:
    if lam is None or lam <= 0:
        return w
    return w / (1.0 + lam * w)


def snis(df: pd.DataFrame, clip: float = 10.0, shrink: float = 0.0) -> dict[str, float]:
    a = df["action"].to_numpy(dtype=float)
    r = df["r"].to_numpy(dtype=float)
    b = df["b_prob"].to_numpy(dtype=float)
    p = df["pi_prob"].to_numpy(dtype=float)
    # One-step ratio for chosen action vs no action under binary actions
    # π(a|s) = p^a (1-p)^(1-a), same for b
    pi_a = p * a + (1 - p) * (1 - a)
    b_a = b * a + (1 - b) * (1 - a)
    w = np.divide(pi_a, b_a, out=np.zeros_like(pi_a), where=b_a > 0)
    w = _clip(w, clip)
    w = _shrink(w, shrink)
    sw = w.sum()
    val = np.divide((w * r).sum(), sw, out=np.array([0.0]), where=sw > 0).item()
    ess = (sw**2) / np.maximum((w**2).sum(), 1e-12)
    return {"value": float(val), "ess": float(ess), "sum_w": float(sw)}


def _select_numeric_features(df: pd.DataFrame, drop: Sequence[str]) -> np.ndarray:
    """Return numeric feature matrix with drops applied and NaNs filled to 0.

    We prefer a simple, stable mapping for OPE diagnostics over exact modeling;
    fill any NaNs/Infs with zeros to avoid numerical blow-ups.
    """
    num_df = df.select_dtypes(include=[np.number]).copy()
    for col in drop:
        if col in num_df.columns:
            num_df.drop(columns=[col], inplace=True)
    arr = num_df.to_numpy(dtype=float)
    # Replace non-finite with zeros to be robust
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _fit_outcome(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Stable ridge regression for E[r | s, a=1].

    - Drops non-finite rows
    - Uses small ridge and linear solve; falls back to zeros if ill-posed
    """
    mask_a1 = df["action"].to_numpy(dtype=bool)
    if mask_a1.sum() < 5:
        return {"coef": np.zeros(1), "intercept": np.array([0.0])}
    X_full = _select_numeric_features(df, drop=["action", "r", "b_prob", "pi_prob"])
    X = X_full[mask_a1]
    y = df.loc[mask_a1, "r"].to_numpy(dtype=float)
    # Keep only finite rows
    finite = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X = X[finite]
    y = y[finite]
    if X.shape[0] < 5:
        return {"coef": np.zeros(1), "intercept": np.array([0.0])}
    # Add intercept and solve ridge
    Xd = np.hstack([np.ones((X.shape[0], 1)), X])
    lam = 1e-2
    XtX = Xd.T @ Xd + lam * np.eye(Xd.shape[1])
    Xty = Xd.T @ y
    try:
        beta = np.linalg.solve(XtX, Xty)
    except Exception:
        return {"coef": np.zeros(1), "intercept": np.array([0.0])}
    return {"coef": beta[1:], "intercept": np.array([beta[0]])}


def dr(df: pd.DataFrame, clip: float = 10.0, shrink: float = 0.0) -> dict[str, float]:
    a = df["action"].to_numpy(dtype=float)
    r = df["r"].to_numpy(dtype=float)
    b = df["b_prob"].to_numpy(dtype=float)
    p = df["pi_prob"].to_numpy(dtype=float)
    # Outcome model for a=1
    pars = _fit_outcome(df)
    X_full = _select_numeric_features(df, drop=["action", "r", "b_prob", "pi_prob"])
    q1 = pars["intercept"][0] + (X_full @ pars["coef"])
    q1 = np.nan_to_num(q1, nan=0.0, posinf=0.0, neginf=0.0)
    # Policy value under model: E[Q(s,π(s))] ≈ E[p*q1 + (1-p)*0]
    v_model = np.mean(p * q1)
    # Importance term
    pi_a = p * a + (1 - p) * (1 - a)
    b_a = b * a + (1 - b) * (1 - a)
    w = np.divide(pi_a, b_a, out=np.zeros_like(pi_a), where=b_a > 0)
    w = _clip(w, clip)
    w = _shrink(w, shrink)
    # DR correction: E[ w * (r - Q(s,a)) ] with Q(s,0)=0
    q_sa = q1 * a
    corr = np.mean(w * (r - q_sa))
    val = float(v_model + corr)
    alpha = float(pars["intercept"][0])
    beta0 = float(pars["coef"][0]) if pars["coef"].size > 0 else 0.0
    return {
        "value": val,
        "intercept": alpha,
        "coef": pars["coef"].tolist(),
        "alpha": alpha,
        "beta": beta0,
    }


__all__ = ["snis", "dr"]
