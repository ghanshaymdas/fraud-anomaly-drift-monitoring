# Fraud / Anomaly Detection with Drift Monitoring

A reproducible Python project for detecting fraudulent transactions and monitoring whether transaction data changes enough to justify model retraining.

## Project goals

This project implements the requirements from the assignment:

- Handle severe fraud/non-fraud class imbalance with **class-weighted supervised learning**.
- Train a **supervised Logistic Regression** fraud classifier.
- Train an **unsupervised Isolation Forest** anomaly detector.
- Compare models using **Precision-Recall AUC (PR-AUC)** rather than relying only on ROC-AUC.
- Monitor drift between a reference/training window and a later transaction window.
- Raise a retraining alert when drift crosses a configurable threshold.
- Analyze false-positive vs false-negative costs and select a classification threshold accordingly.
- Demonstrate drift detection with a reproducible synthetic drift smoke test.

## Repository structure

```text
fraud-anomaly-drift-monitoring/
├── README.md
├── requirements.txt
├── .gitignore
├── run_pipeline.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── drift.py
│   └── evaluation.py
├── tests/
│   └── smoke_test.py
├── data/
│   └── README.md
└── results/
    └── README.md
```

## Dataset

The intended dataset is the **Credit Card Fraud Detection** dataset, commonly distributed as `creditcard.csv`.

Download it from Kaggle and place it here:

```text
data/creditcard.csv
```

The program does **not** download the dataset automatically, because Kaggle credentials and dataset licenses vary.

The expected target column is:

```text
Class
```

where `0 = legitimate` and `1 = fraud`.

The program also works with the common Kaggle column names such as `Time`, `Amount`, and `V1` ... `V28`.

## Why class weighting?

Fraud is an extremely rare class. Naively training a normal classifier can produce a model that looks accurate while missing fraud.

Instead, the supervised model uses:

```python
class_weight="balanced"
```

This increases the training penalty for mistakes on the minority fraud class without creating synthetic examples. It is simple, reproducible, and appropriate as a baseline.

## Models

### 1. Supervised model

A scaled Logistic Regression classifier is used as a transparent baseline.

Pipeline:

```text
transaction features
        ↓
StandardScaler
        ↓
LogisticRegression(class_weight="balanced")
        ↓
fraud probability
```

### 2. Unsupervised model

Isolation Forest is trained without using fraud labels.

```text
transaction features
        ↓
Isolation Forest
        ↓
anomaly score
```

The anomaly score is converted into a comparable ranking score where larger values mean "more suspicious".

## Evaluation

Because fraud is highly imbalanced, the main ranking metric is **Average Precision / PR-AUC**.

The project also reports:

- precision
- recall
- F1
- confusion matrix
- chosen classification threshold

ROC-AUC can be misleading in highly imbalanced problems, so it is intentionally not the headline metric.

## Threshold and cost analysis

Blocking a legitimate transaction is a false positive. Missing a fraudulent transaction is a false negative.

The default example costs are:

```text
false-positive cost = 1
false-negative cost = 20
```

These are configurable rather than claimed to be universal business values.

The threshold-selection routine searches thresholds on the validation set and minimizes:

```text
expected cost =
    FP × false_positive_cost
  + FN × false_negative_cost
```

while also requiring a minimum recall target when possible.

In a real fintech deployment, these costs should be replaced by measured business costs.

## Drift monitoring

The monitor compares a reference window with a later window using:

1. **Population Stability Index (PSI)** for numerical feature distributions.
2. A change in the model's predicted fraud probability distribution.

A feature is considered materially shifted when its PSI exceeds the configured feature threshold. The overall monitor raises a retraining alert when the fraction of monitored features with high PSI exceeds the configured alert threshold, or when confidence drift is high.

PSI interpretation is treated as a practical monitoring heuristic, not a universal law.

## Simulated drift bonus

Run:

```bash
python run_pipeline.py --smoke-test
```

This generates synthetic transaction data, trains the models, creates a shifted later window, and demonstrates that the drift monitor detects the change.

No Kaggle dataset is required for the smoke test.

## Run with the real dataset

Create an environment and install dependencies:

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Then place `creditcard.csv` in `data/` and run:

```bash
python run_pipeline.py
```

Optional:

```bash
python run_pipeline.py --csv data/creditcard.csv --test-size 0.25
```

## Reproducibility

The project uses fixed random seeds where randomness is involved. The exact metrics depend on the dataset version and the train/test split.

Do not claim a specific result in a report until the pipeline has been run on the actual downloaded dataset.

## GitHub submission

From the project directory:

```bash
git init
git add .
git commit -m "Add fraud anomaly detection with drift monitoring"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fraud-anomaly-drift-monitoring.git
git push -u origin main
```

Make the GitHub repository **Public** before submitting its URL.

## Suggested write-up structure

1. Problem and motivation
2. Data and class imbalance
3. Supervised model
4. Unsupervised anomaly detection
5. PR-AUC comparison
6. Threshold/cost analysis
7. Drift monitoring
8. Simulated drift result
9. Limitations and future improvements

## Limitations

This is an educational baseline, not a production fraud system. Real deployment would require temporal validation, leakage checks, transaction/customer-level features, calibration, monitoring of delayed fraud labels, fairness analysis, investigation workflows, and business-specific cost estimates.
