# Topic 11 — Neural Networks, SVMs & Transformers

Three short tasks that each use a different modern model family:

1. **task1** — a **neural network** (multi-layer perceptron) that predicts a number.
2. **task2a** — a **Support Vector Machine** that recognises handwritten digits.
3. **task2b** — a **pretrained transformer** that reads text and judges its sentiment.

---

## Task 1 — MLP Regression

**Goal:** predict a continuous target `y` from a table of numeric features.

**Pipeline:** load `lab4_data.csv` → fill missing values → keep the 10 most
relevant features with `SelectKBest` → scale → train an `MLPRegressor` with three
hidden layers (64 → 32 → 16 neurons, ReLU activation) → score on a held-out 20%.

### Result (on the synthetic dataset below)

| Metric | Value | Plain meaning                                                        |
|--------|-------|---------------------------------------------------------------------|
| R²     | 0.983 | The model explains ~98% of the variation in `y` — an excellent fit. |
| RMSE   | 17.23 | Typical prediction error, in the same units as `y`.                 |
| MAE    | 13.53 | Average absolute error (less sensitive to big misses than RMSE).    |

### Note on the data

The original lab dataset wasn't included, so `make_dataset.py` builds a
**synthetic stand-in** (`lab4_data.csv`) with the exact shape `task1.py` expects:
15 numeric feature columns plus a numeric target `y`. Only **6 of the 15 features
truly influence `y`** (the rest are noise) and a few values are left blank on
purpose — so the feature-selection and missing-value steps actually do something.
Regenerate it any time with:

```bash
python3 make_dataset.py
```

### Plots (in [`plots/`](plots/))

| File            | What it shows                          | How to read it                                                                                                                                                              |
|-----------------|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `figure_01.png` | **Training loss curve**                | Loss (prediction error) on the y-axis vs. training iteration on the x-axis. It starts high and drops steeply, then flattens — that flattening means the network has *converged* (extra training would barely help). A curve that never flattens would mean it needs more iterations; one that dips then rises would hint at trouble. |
| `figure_02.png` | **Predicted vs. actual** scatter       | Each dot is one test home: its true `y` (x-axis) against the model's guess (y-axis). The red dashed line is "perfect". Because the dots hug that line tightly, the model is accurate; dots far above the line are over-predictions, far below are under-predictions. The tight cloud matches the high R² of 0.98. |

---

## Task 2a — SVM on Handwritten Digits

**Goal:** classify 8×8 pixel images of digits (0–9) from scikit-learn's `digits`
dataset using a **Support Vector Machine** with an RBF kernel.

**Result:** **~98% accuracy** on the test set. The per-class precision and recall
(printed by the script) are all ≥ 0.94, meaning the SVM rarely confuses one digit
for another — impressive for such tiny, low-resolution images. An SVM works well
here because it finds the widest possible margin between classes, which
generalises cleanly even in the high-dimensional (64-pixel) space.

*(This task prints a classification report to the console; it doesn't save a plot.)*

---

## Task 2b — Transformer Sentiment Analysis

**Goal:** feed a few sentences to a **pretrained transformer** (via Hugging Face
`transformers.pipeline("sentiment-analysis")`) and have it label each as POSITIVE
or NEGATIVE with a confidence score — no training required, the model already
knows language.

> **Not run in this environment.** The `pipeline` needs a deep-learning backend
> (**PyTorch**, ~800 MB) plus a one-time model download from Hugging Face, neither
> of which is installed here. To run it yourself:
>
> ```bash
> pip install torch            # CPU build is fine
> python3 task2b.py
> ```
>
> Expected behaviour: the two clearly positive/negative sentences get labelled
> with high confidence (~0.99), while the neutral "It's okay, nothing special."
> gets a lower, less-confident score.

---

## Run everything

```bash
python3 make_dataset.py   # (re)build the synthetic dataset
python3 task1.py          # MLP regression + plots
python3 task2a.py         # SVM on digits
python3 task2b.py         # sentiment (needs torch — see above)
```
