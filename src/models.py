from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_supervised_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_isolation_forest(random_state: int = 42) -> IsolationForest:
    return IsolationForest(
        n_estimators=250,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )


def train_models(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    supervised = build_supervised_model(random_state)
    supervised.fit(X_train, y_train)

    anomaly = build_isolation_forest(random_state)
    anomaly.fit(X_train)

    return supervised, anomaly


def supervised_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def anomaly_scores(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    # Isolation Forest: lower decision_function = more anomalous.
    # Negating it makes larger values more suspicious.
    return -model.decision_function(X)
