from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def pr_auc(y_true, scores) -> float:
    return float(average_precision_score(y_true, scores))


def threshold_metrics(y_true, scores, threshold: float) -> dict:
    predictions = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def choose_cost_sensitive_threshold(
    y_true,
    scores,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 20.0,
    min_recall: float = 0.80,
) -> dict:
    scores = np.asarray(scores)
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99),
                np.quantile(scores, np.linspace(0.01, 0.99, 99)),
            ]
        )
    )
    thresholds = np.clip(thresholds, 0.000001, 0.999999)

    candidates = []
    for threshold in thresholds:
        metrics = threshold_metrics(y_true, scores, float(threshold))
        cost = (
            metrics["false_positives"] * false_positive_cost
            + metrics["false_negatives"] * false_negative_cost
        )
        metrics["cost"] = float(cost)
        candidates.append(metrics)

    eligible = [m for m in candidates if m["recall"] >= min_recall]
    pool = eligible if eligible else candidates
    return min(pool, key=lambda m: (m["cost"], -m["f1"]))
