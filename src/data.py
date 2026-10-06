from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


def load_credit_card_csv(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download creditcard.csv from Kaggle "
            "and place it at data/creditcard.csv."
        )

    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Expected a binary target column named 'Class'.")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])

    if y.nunique() != 2:
        raise ValueError("The Class column must contain both 0 and 1.")
    return X, y


def make_smoke_data(
    n_samples: int = 12000,
    n_features: int = 12,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.985, 0.015],
        class_sep=1.4,
        random_state=random_state,
    )
    columns = [f"feature_{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=columns), pd.Series(y, name="Class")


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
