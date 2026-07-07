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

### `figure_01.png` — Oil vs Temperature and Oil vs Insulation (two scatter plots)

**What it shows:** two side-by-side scatter plots. Left: each home's outside
temperature (x) against its oil use (y). Right: each home's insulation (x)
against its oil use (y).

**How to read it:** both clouds of dots slope **downward** from left to right —
as temperature rises, oil use falls, and as insulation thickens, oil use falls
too. That downward slope is the visual version of the two **negative
coefficients** in the equation. The tighter and straighter the band of dots, the
stronger that relationship — which is why a simple straight-line model fits so
well here (R² = 0.97).

### `figure_02.png` — Putt success rate with the fitted logistic curve

**What it shows:** red dots are the **actual** make-rate observed at each putt
length; the blue S-shaped line is the model's **predicted** probability of making
the putt.

**How to read it:** the curve slides downhill — short putts sit high (likely
made), long putts sit low (likely missed). Because the red dots fall close to the
blue curve, the model captures the real pattern well. The characteristic
**S-shape** (flat near the top and bottom, steep in the middle) is what makes
logistic regression different from a straight line: probabilities can never go
above 1 or below 0, so the curve levels off at both ends instead of shooting past
them.

## Run it

```bash
python3 Topic5_Regression.py
```

Plots are saved to `plots/`.
