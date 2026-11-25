import numpy as np
import pandas as pd

FEATURES = ["tenure", "monthly_charge", "support_calls", "contract"]

def sample_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({"tenure": rng.integers(1, 73, n), "monthly_charge": rng.uniform(20, 120, n).round(2), "support_calls": rng.integers(0, 8, n), "contract": rng.choice(["monthly", "annual"], n)})
    logits = -1 - .035*data.tenure + .018*data.monthly_charge + .28*data.support_calls + .8*(data.contract == "monthly")
    data["churn"] = rng.binomial(1, 1/(1+np.exp(-logits)))
    return data

def validate(data):
    missing = set(FEATURES + ["churn"]) - set(data.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    data = data[FEATURES + ["churn"]].copy()
    for name in FEATURES[:3] + ["churn"]:
        data[name] = pd.to_numeric(data[name], errors="raise")
    if data.churn.isna().any() or set(data.churn.unique()) != {0, 1} or data.churn.value_counts().min() < 5:
        raise ValueError("Need at least five examples of each binary churn class")
    return data
