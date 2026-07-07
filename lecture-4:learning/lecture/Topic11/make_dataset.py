"""
Generates a synthetic regression dataset (lab4_data.csv) for task1.py.

The original lab dataset wasn't included, so this builds a stand-in with the
same shape task1.py expects: several numeric feature columns plus a numeric
target column named "y". Only a handful of features actually drive the target
(the rest are noise), which lets SelectKBest show off feature selection.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

# Reproducible synthetic data
X, y = make_regression(
    n_samples=1000,
    n_features=15,      # 15 candidate features...
    n_informative=6,    # ...only 6 truly influence y (the rest are noise)
    noise=12.0,         # realistic measurement noise
    random_state=42,
)

cols = [f"x{i+1}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=cols)
df["y"] = y

# Sprinkle a few missing values so the fillna step in task1.py is exercised
rng = np.random.default_rng(42)
for c in rng.choice(cols, size=3, replace=False):
    idx = rng.choice(df.index, size=20, replace=False)
    df.loc[idx, c] = np.nan

df.to_csv("lab4_data.csv", index=False)
print(f"Wrote lab4_data.csv with shape {df.shape} "
      f"({X.shape[1]} features, 6 informative, target='y')")
