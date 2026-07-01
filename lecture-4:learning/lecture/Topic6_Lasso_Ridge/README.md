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

| File            | What it shows                                                            |
|-----------------|--------------------------------------------------------------------------|
| `figure_01.png` | Coefficients of the Lasso-selected features (non-zero = kept).           |
| `figure_02.png` | Ridge predictions vs actual values; points on the dashed line = perfect. |

## Note on the data

`data.csv` is a small **synthetic sample dataset** generated for this lab (the
original wasn't included), where the target depends on only a few features — so
the results are illustrative.

## Run it

```bash
python3 Topic6_Lasso_Ridge.py
```

Plots are saved to `plots/`.
