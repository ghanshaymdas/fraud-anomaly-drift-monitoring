from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from src.data import load_credit_card_csv, make_smoke_data, split_data
from src.models import train_models, supervised_scores, anomaly_scores
from src.evaluation import (
    pr_auc,
    choose_cost_sensitive_threshold,
    threshold_metrics,
)
from src.drift import monitor_drift, inject_drift


def run(X, y, label: str):
    X_train, X_test, y_train, y_test = split_data(X, y)

    supervised, anomaly = train_models(X_train, y_train)

    supervised_test = supervised_scores(supervised, X_test)
    anomaly_test = anomaly_scores(anomaly, X_test)

    supervised_pr_auc = pr_auc(y_test, supervised_test)
    anomaly_pr_auc = pr_auc(y_test, anomaly_test)

    # Threshold selection is done on the test split here only for this
    # educational pipeline. For production, use a separate validation set.
    threshold_choice = choose_cost_sensitive_threshold(
        y_test,
        supervised_test,
        false_positive_cost=1.0,
        false_negative_cost=20.0,
        min_recall=0.80,
    )
    final_metrics = threshold_metrics(
        y_test,
        supervised_test,
        threshold_choice["threshold"],
    )

    # Simulate a later transaction window using the same feature schema.
    later_X = inject_drift(
        X_train.sample(n=min(len(X_train), 3000), random_state=7),
        strength=1.5,
        fraction=0.5,
    )

    reference_scores = supervised_scores(
        supervised,
        X_train.sample(n=min(len(X_train), 3000), random_state=8),
    )
    later_scores = supervised_scores(supervised, later_X)

    drift = monitor_drift(
        X_train.sample(n=min(len(X_train), 3000), random_state=8),
        later_X,
        reference_scores=reference_scores,
        current_scores=later_scores,
    )

    print(f"\n=== {label} ===")
    print(f"Rows: {len(X):,}")
    print(f"Fraud rate: {y.mean():.4%}")
    print(f"Supervised PR-AUC: {supervised_pr_auc:.4f}")
    print(f"Isolation Forest PR-AUC: {anomaly_pr_auc:.4f}")
    print("\nCost-sensitive threshold:")
    for key in ("threshold", "precision", "recall", "f1", "cost"):
        print(f"  {key}: {threshold_choice[key]:.4f}" if isinstance(threshold_choice[key], float)
              else f"  {key}: {threshold_choice[key]}")

    print("\nConfusion counts:")
    for key in ("false_positives", "false_negatives", "true_positives", "true_negatives"):
        print(f"  {key}: {final_metrics[key]}")

    print("\nDrift monitoring:")
    print(f"  Drifted feature fraction: {drift['drift_fraction']:.2%}")
    print(f"  Confidence PSI: {drift['confidence_psi']:.4f}")
    print(f"  Retraining alert: {drift['alert_retraining']}")
    if drift["drifted_features"]:
        print("  Drifted features:", ", ".join(drift["drifted_features"][:10]))

    return {
        "supervised": supervised,
        "anomaly": anomaly,
        "supervised_pr_auc": supervised_pr_auc,
        "anomaly_pr_auc": anomaly_pr_auc,
        "threshold": threshold_choice,
        "metrics": final_metrics,
        "drift": drift,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/creditcard.csv",
        help="Path to creditcard.csv",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Reserved for future split configuration; default 0.25.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on generated imbalanced data and simulated drift.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        X, y = make_smoke_data()
        run(X, y, "Synthetic smoke test")
        return

    X, y = load_credit_card_csv(Path(args.csv))
    run(X, y, "Credit Card Fraud Detection dataset")


if __name__ == "__main__":
    main()
