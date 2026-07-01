# Heart Disease Prediction — Stacking Ensemble

A small end-to-end machine-learning project that predicts the presence of heart
disease from clinical measurements using a stacked ensemble of classifiers.

## What it does

1. **Loads data** — pulls the UCI Heart Disease dataset from Hugging Face
   (`skrishna/heart_disease_uci`), with a CSV fallback if that is unavailable.
2. **Cleans it** — drops ID columns, binarizes the target (0 = no disease,
   1 = disease), and trims string values.
3. **Explores it** — generates class-balance, correlation, and feature-distribution plots.
4. **Builds a pipeline** — imputation → scaling / one-hot encoding →
   `SelectKBest` feature selection → a **stacking classifier**.
5. **Evaluates** — reports accuracy, precision, recall, F1, ROC-AUC and 5-fold
   cross-validation, plus diagnostic plots.

## The model

A `StackingClassifier` combining three base learners whose predictions are
combined by a logistic-regression meta-learner:

| Base estimator        | Role                        |
|-----------------------|-----------------------------|
| Random Forest         | bagged decision trees       |
| Gradient Boosting     | boosted decision trees      |
| SVC (RBF kernel)      | margin-based classifier     |
| → Logistic Regression | final meta-estimator        |

## How the pipeline works

Everything the model does — from raw data to a prediction — is wrapped in a
single scikit-learn `Pipeline`. This matters because the **exact same steps are
applied to training and test data automatically**, so there is no risk of
accidentally "leaking" test information into training. The data flows through
these stages in order:

```
Raw patient data
      │
      ▼
1. Preprocessing  ──►  Numeric columns:  fill missing values (median) → scale to a common range
                       Category columns: fill missing values (most frequent) → one-hot encode
      │
      ▼
2. Feature selection (SelectKBest)  ──►  keep only the 10 most informative features
      │
      ▼
3. Stacking classifier
        ├─ Random Forest      ┐
        ├─ Gradient Boosting  ├─►  each makes its own prediction
        └─ SVC (RBF)          ┘
                 │
                 ▼
        Logistic Regression   ──►  learns how to best combine the three
                 │
                 ▼
            Final prediction  (disease / no disease + probability)
```

In short: **clean the data → keep the best features → let three different
models vote → a final model learns the smartest way to combine those votes.**
The whole pipeline is validated with **5-fold cross-validation**, meaning the
data is split into 5 parts and the model is trained and tested 5 times on
different slices, so the score isn't a lucky one-off.

## Model performance

On the held-out test set (20% of the data the model never saw during training):

| Metric      | Score | What it means (in plain terms)                                  |
|-------------|-------|-----------------------------------------------------------------|
| Accuracy    | 0.84  | 84% of all patients were classified correctly.                  |
| Precision   | 0.78  | When it predicts "disease", it's right about 78% of the time.   |
| Recall      | 0.89  | It catches 89% of patients who actually have the disease.       |
| F1-score    | 0.83  | Balance between precision and recall.                           |
| ROC-AUC     | 0.94  | 94% chance it ranks a sick patient above a healthy one.         |

Cross-validation accuracy: **0.83 ± 0.03**, which is very close to the test-set
accuracy — a good sign that the model is **stable and not overfitting**.

**Reading between the numbers:** recall (0.89) is higher than precision (0.78),
which is exactly what you want in a medical screening tool — it would rather
raise a few false alarms than miss a patient who genuinely has heart disease.
The high ROC-AUC (0.94) shows the model is very good at separating the two
groups overall.

## The plots and what each one is for

All generated diagrams are written to the [`plots/`](plots/) folder. They fall
into two groups — plots that help you **understand the data before modelling**,
and plots that help you **judge the model after training**.

### Understanding the data

| File                          | Purpose in the pipeline                                                                 |
|-------------------------------|-----------------------------------------------------------------------------------------|
| `target_distribution.png`     | Shows how many patients have vs. don't have disease — checks the classes are balanced.  |
| `correlation_heatmap.png`     | Shows which features move together and which relate most to the diagnosis.              |
| `feature_distributions.png`   | Histograms of each numeric feature — reveals their range, skew, and outliers.           |

### Judging the model

| File                          | Purpose in the pipeline                                                                 |
|-------------------------------|-----------------------------------------------------------------------------------------|
| `confusion_matrix.png`        | Counts correct vs. wrong predictions, split into true/false positives and negatives.    |
| `roc_curve.png`               | Shows the trade-off between catching disease and false alarms; area under it = ROC-AUC.  |
| `precision_recall_curve.png`  | Focuses on the "disease" class — useful when catching positives matters most.           |
| `feature_importance.png`      | Ranks which features the Random Forest relied on most to make decisions.                 |
| `cross_validation_scores.png` | Accuracy on each of the 5 folds — shows how consistent the model is across data splits.  |

## Running it

```bash
python3 project.py
```

Required packages (`datasets`, `scikit-learn`, `seaborn`, plus numpy / pandas /
matplotlib) are auto-installed on first run. Metrics print to the console and all
plots are saved to `plots/`.
