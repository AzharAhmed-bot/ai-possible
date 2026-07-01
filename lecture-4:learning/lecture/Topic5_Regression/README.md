# Topic 5 — Regression Models

Two classic regression tasks: predicting a **number** (linear regression) and
predicting a **probability** (logistic regression).

## Q1 — Multiple Linear Regression: Heating Oil

**Goal:** predict how much heating oil a home uses (gallons) from the outside
**temperature** and the amount of **insulation**.

- Both predictors have a **negative** effect: warmer weather and thicker
  insulation → less oil used.
- Learned equation:
  `Oil = 562.15 + (−5.44 × Temp) + (−20.01 × Insulation)`
- Fit quality: **R² = 0.97** (the model explains 97% of the variation).
- Example prediction — Temp 15 °F, Insulation 10 in → **≈ 280 gallons**.

## Q2 — Logistic Regression: Golf Putts

**Goal:** predict the **chance of making a putt** from its length (feet).

- The longer the putt, the lower the success probability — a smooth S-shaped
  (logistic) curve.
- Learned logit: `log(p/(1−p)) = 3.24 − 0.56 × Length`.
- Predicted make-rate drops from ~73% at 4 ft to ~33% at 7 ft.

## The difference (in plain terms)

- **Linear regression** predicts a number on a continuous scale (gallons).
- **Logistic regression** predicts a probability between 0 and 1 (made / missed).

## Plots (in [`plots/`](plots/))

| File            | What it shows                                                              |
|-----------------|---------------------------------------------------------------------------|
| `figure_01.png` | Scatter plots of oil vs temperature and oil vs insulation (the raw trends). |
| `figure_02.png` | Observed putt success rates with the fitted logistic probability curve.   |

## Run it

```bash
python3 Topic5_Regression.py
```

Plots are saved to `plots/`.
