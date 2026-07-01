# Topic 9 — Loans: Ensemble vs Perceptron

Predicts whether a loan will be **fully repaid** or **not**, comparing two very
different classifiers: a **Random Forest** (an ensemble of decision trees) and a
**Perceptron** (the simplest one-layer neural network).

## What it does

1. Loads `loan_data.csv`, encodes the categorical `purpose` column, and splits
   the data 80/20 (keeping the class balance with `stratify`).
2. **Scales** the features (needed for the Perceptron).
3. Trains a **Random Forest** (200 trees) and a **Perceptron**, then compares them
   on accuracy, AUC-ROC and confusion matrices.

## The two models (in plain terms)

- **Random Forest** — grows many decision trees on random slices of the data and
  lets them vote. Captures complex, non-linear patterns and rarely overfits.
- **Perceptron** — draws a single straight dividing line. Fast, but can't capture
  non-linear patterns, so it usually loses to the forest.

## Result

On this sample data the **Random Forest** comes out ahead
(accuracy ≈ 0.55 vs the Perceptron's ≈ 0.49), which matches the theory — an
ensemble beats a single linear classifier.

## Plots (in [`plots/`](plots/))

| File            | What it shows                                                            |
|-----------------|--------------------------------------------------------------------------|
| `figure_01.png` | Class balance — how many loans were paid vs not fully paid.              |
| `figure_02.png` | Random Forest confusion matrix + which features mattered most.           |
| `figure_03.png` | Perceptron confusion matrix.                                             |
| `figure_04.png` | Accuracy comparison of the two models.                                   |
| `figure_05.png` | ROC curve for the Random Forest (higher AUC = better separation).        |

## Note on the data

`loan_data.csv` is a **synthetic sample dataset** in the LendingClub style,
generated for this lab because the original file wasn't included. The target
signal is deliberately weak, so the accuracy numbers are illustrative — the point
is the *method and comparison*, not the exact scores.

## Run it

```bash
python3 Topic9_Loans_Ensemble_Perceptron.py
```

Plots are saved to `plots/`.
