# Topic 6 — Lasso & Ridge Regression

Shows how **regularization** helps regression: **Lasso** picks the important
features, then **Ridge** builds a stable model on just those features.

## What it does

1. Loads `data.csv` (features + a target in the last column) and splits it 80/20.
2. **Scales** the features (required so the penalties treat them fairly).
3. **Lasso (L1)** — shrinks unhelpful feature coefficients to *exactly zero*, so
   it doubles as automatic **feature selection**.
4. **Ridge (L2)** — refits on the selected features, gently shrinking
   coefficients to avoid overfitting.
5. Reports RMSE and R² on the test set.

## The two penalties (in plain terms)

- **Lasso** can switch features off completely → a shorter, simpler model.
- **Ridge** keeps all features but keeps their weights small → a steadier model.

## Result

- Features kept by Lasso: **7 of 8**.
- Ridge test performance: **RMSE = 0.53**, **R² = 0.99** (an excellent fit).

## Plots (in [`plots/`](plots/))

### `figure_01.png` — Lasso feature coefficients (bar chart)

**What it shows:** one bar per feature, its height (and direction) being the
**weight** Lasso gave that feature.

**How to read it:** a **taller** bar = a feature with a bigger say in the
prediction; bars **above zero** push the target up, bars **below zero** push it
down. Any feature Lasso judged useless has been shrunk to a **zero-height bar** —
that's the "automatic feature selection" in action (here Lasso kept 7 of 8). In
short: the surviving bars are the features that actually matter, and their size
tells you how much.

### `figure_02.png` — Ridge: predicted vs actual (scatter)

**What it shows:** each dot is one test row — its true value (x-axis) against the
model's prediction (y-axis). The red dashed line is the "perfect prediction" line
(predicted = actual).

**How to read it:** dots that sit **right on the dashed line** were predicted
almost exactly; dots **above** the line were over-predicted, dots **below** were
under-predicted. Here the dots hug the line tightly across the whole range, which
is the visual proof behind the excellent **R² = 0.99** — the model is accurate
and isn't biased high or low at any part of the scale.

## Note on the data

`data.csv` is a small **synthetic sample dataset** generated for this lab (the
original wasn't included), where the target depends on only a few features — so
the results are illustrative.

## Run it

```bash
python3 Topic6_Lasso_Ridge.py
```

Plots are saved to `plots/`.
