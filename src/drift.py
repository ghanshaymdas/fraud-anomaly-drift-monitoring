from __future__ import annotations

import numpy as np
import pandas as pd


def _bin_edges(reference: np.ndarray, bins: int = 10) -> np.ndarray:
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        lo, hi = float(np.min(reference)), float(np.max(reference))
        if lo == hi:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def psi(reference, current, bins: int = 10, epsilon: float = 1e-6) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    edges = _bin_edges(reference, bins=bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / max(len(reference), 1), epsilon, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), epsilon, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_psi_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    bins: int = 10,
) -> pd.Series:
    common = [c for c in reference.columns if c in current.columns]
    values = {
        c: psi(reference[c].to_numpy(), current[c].to_numpy(), bins=bins)
        for c in common
        if pd.api.types.is_numeric_dtype(reference[c])
    }
    return pd.Series(values, dtype=float).sort_values(ascending=False)


def monitor_drift(
    reference_X: pd.DataFrame,
    current_X: pd.DataFrame,
    reference_scores=None,
    current_scores=None,
    feature_psi_threshold: float = 0.20,
    alert_feature_fraction: float = 0.30,
    confidence_psi_threshold: float = 0.20,
) -> dict:
    feature_psis = feature_psi_report(reference_X, current_X)
    drifted = feature_psis >= feature_psi_threshold
    drift_fraction = float(drifted.mean()) if len(drifted) else 0.0

    confidence_psi = None
    if reference_scores is not None and current_scores is not None:
        confidence_psi = psi(
            np.asarray(reference_scores),
            np.asarray(current_scores),
        )

    alert = (
        drift_fraction >= alert_feature_fraction
        or (
            confidence_psi is not None
            and confidence_psi >= confidence_psi_threshold
        )
    )

    return {
        "feature_psis": feature_psis,
        "drifted_features": list(feature_psis.index[drifted]),
        "drift_fraction": drift_fraction,
        "confidence_psi": confidence_psi,
        "alert_retraining": bool(alert),
    }


def inject_drift(
    X: pd.DataFrame,
    strength: float = 1.5,
    fraction: float = 0.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a simulated later window by shifting numeric features."""
    rng = np.random.default_rng(random_state)
    shifted = X.copy()
    numeric = list(shifted.select_dtypes(include=np.number).columns)
    count = max(1, int(len(numeric) * fraction))

    for column in numeric[:count]:
        std = float(shifted[column].std())
        shifted[column] = shifted[column] + strength * std + rng.normal(
            0, 0.05 * max(std, 1e-9), size=len(shifted)
        )

    return shifted
