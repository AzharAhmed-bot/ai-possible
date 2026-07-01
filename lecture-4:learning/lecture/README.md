# Lecture 4 — Learning (Labs & Project)

Machine-learning labs and a final project. Each folder is self-contained: it has
its Python script, a `plots/` folder with the figures it generates, and a
`README.md` explaining what it does.

## Contents

| Folder | Topic | What you'll find |
|--------|-------|------------------|
| [`Topic2_Visualization/`](Topic2_Visualization/) | Data visualization | 8 matplotlib chart types and when to use each. |
| [`Topic5_Regression/`](Topic5_Regression/) | Regression | Linear regression (predict a number) + logistic regression (predict a probability). |
| [`Topic6_Lasso_Ridge/`](Topic6_Lasso_Ridge/) | Regularization | Lasso feature selection followed by Ridge regression. |
| [`Topic9_Loans_Ensemble_Perceptron/`](Topic9_Loans_Ensemble_Perceptron/) | Classification | Random Forest vs Perceptron on loan-repayment prediction. |
| [`Lab3_Q_Learning/`](Lab3_Q_Learning/) | Reinforcement learning | A Q-learning agent that solves a small grid world. |
| [`project/`](project/) | **Project** | Heart-disease prediction with a stacking ensemble (full write-up inside). |

## Running any of them

Each script is run from inside its own folder so its data and `plots/` resolve
correctly:

```bash
cd Topic5_Regression
python3 Topic5_Regression.py
```

Required packages (numpy, pandas, matplotlib, seaborn, scikit-learn, and for the
project `datasets`) install automatically or via `pip`.

## Note on data

Two labs (`Topic6_Lasso_Ridge` and `Topic9_Loans_Ensemble_Perceptron`) ship with
small **synthetic sample datasets**, since their original CSVs weren't included.
The code is unchanged — only the input data is illustrative.
