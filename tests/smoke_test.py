from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import make_smoke_data
from src.models import train_models, supervised_scores
from src.evaluation import pr_auc
from src.drift import monitor_drift, inject_drift


def main():
    X, y = make_smoke_data(n_samples=4000)
    X_train = X.iloc[:2800]
    y_train = y.iloc[:2800]
    X_later = inject_drift(X.iloc[2800:], strength=2.0, fraction=0.6)

    supervised, _ = train_models(X_train, y_train)

    reference_scores = supervised_scores(supervised, X_train)
    later_scores = supervised_scores(supervised, X_later)

    report = monitor_drift(
        X_train,
        X_later,
        reference_scores=reference_scores,
        current_scores=later_scores,
    )

    assert 0.0 <= pr_auc(y.iloc[2800:], later_scores) <= 1.0
    assert report["drift_fraction"] > 0.0
    assert report["alert_retraining"] is True

    print("Smoke test passed.")
    print(f"Drifted feature fraction: {report['drift_fraction']:.2%}")
    print(f"Confidence PSI: {report['confidence_psi']:.4f}")


if __name__ == "__main__":
    main()
