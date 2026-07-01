import numpy as np
import pandas as pd
import matplotlib as _mpl
_mpl.use("Agg")  # headless backend: save figures to files instead of opening a window
import matplotlib.pyplot as plt
import os as _os
_os.makedirs("plots", exist_ok=True)
_FIG_N = [0]
def _save():
    """Save the current figure to plots/figure_NN.png (replaces plt.show)."""
    _FIG_N[0] += 1
    plt.savefig(_os.path.join("plots", f"figure_{_FIG_N[0]:02d}.png"), dpi=150, bbox_inches="tight")
    plt.close()
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Load the loan dataset — place loan_data.csv in the same folder
df = pd.read_csv('loan_data.csv', index_col=0)

print('Shape:', df.shape)
print('\nFirst 5 rows:')
print(df.head())
print('\nData types:')
print(df.dtypes)
print('\nMissing values:')
print(df.isnull().sum())

# Quick EDA — class balance
print('Target distribution (not_fully_paid):')
print(df['not_fully_paid'].value_counts())

plt.figure(figsize=(6, 4))
df['not_fully_paid'].value_counts().plot(kind='bar', color=['steelblue', 'coral'], edgecolor='black')
plt.title('Loan Repayment Status')
plt.xlabel('0 = Paid Back | 1 = Not Fully Paid')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
_save()

# 'purpose' is categorical — encode it to numeric
le = LabelEncoder()
df['purpose'] = le.fit_transform(df['purpose'])

# Separate features and target
X = df.drop(columns=['not_fully_paid'])
y = df['not_fully_paid']

print('Features:', list(X.columns))
print('\nFeature shape:', X.shape)

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify preserves class ratio
)
print(f'\nTrain: {X_train.shape}, Test: {X_test.shape}')

# Scale features (required for Perceptron; won't hurt RF)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Random Forest is an ensemble of decision trees.
# It builds many trees on random subsets of data/features,
# then aggregates predictions by majority vote (classification).
# n_estimators=200 → 200 trees; more trees = more stable but slower.
# max_depth=10 → limits tree depth to prevent overfitting.
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1           # use all available CPU cores
)
rf.fit(X_train_sc, y_train)   # train on scaled data
rf_pred  = rf.predict(X_test_sc)
rf_proba = rf.predict_proba(X_test_sc)[:, 1]  # probabilities for AUC-ROC

print('=== Random Forest — Evaluation ===')
print(f'Accuracy : {accuracy_score(y_test, rf_pred):.4f}')
print(f'AUC-ROC  : {roc_auc_score(y_test, rf_proba):.4f}')
print()
print(classification_report(y_test, rf_pred, target_names=['Fully Paid', 'Not Fully Paid']))

# Confusion matrix for Random Forest
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Pred: Paid', 'Pred: Not Paid'],
            yticklabels=['True: Paid', 'True: Not Paid'])
axes[0].set_title('Random Forest — Confusion Matrix')

# Feature importance
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
feat_imp.plot(kind='bar', ax=axes[1], color='steelblue', edgecolor='black')
axes[1].set_title('Random Forest — Feature Importance')
axes[1].set_ylabel('Importance')
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
_save()

# Perceptron is the simplest neural network — a single-layer linear classifier.
# It updates weights based on misclassified samples only (no gradient / probabilities).
# max_iter=1000 → maximum training passes through the data.
# eta0=0.1 → learning rate controlling how big each weight update is.
perc = Perceptron(
    max_iter=1000,
    eta0=0.1,           # learning rate
    random_state=42,
    tol=1e-3            # stop early if loss doesn't improve
)
perc.fit(X_train_sc, y_train)
perc_pred = perc.predict(X_test_sc)

print('=== Perceptron — Evaluation ===')
print(f'Accuracy : {accuracy_score(y_test, perc_pred):.4f}')
print()
print(classification_report(y_test, perc_pred, target_names=['Fully Paid', 'Not Fully Paid']))

# Confusion matrix for Perceptron
plt.figure(figsize=(6, 5))
cm_perc = confusion_matrix(y_test, perc_pred)
sns.heatmap(cm_perc, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Pred: Paid', 'Pred: Not Paid'],
            yticklabels=['True: Paid', 'True: Not Paid'])
plt.title('Perceptron — Confusion Matrix')
plt.tight_layout()
_save()

# Side-by-side accuracy comparison
models      = ['Random Forest', 'Perceptron']
accuracies  = [
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, perc_pred)
]

plt.figure(figsize=(7, 5))
bars = plt.bar(models, accuracies, color=['steelblue', 'coral'], edgecolor='black', width=0.4)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=12)
plt.ylim(0, 1.1)
plt.title('Model Accuracy Comparison', fontsize=14)
plt.ylabel('Accuracy')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
_save()

print('\nSummary:')
print(f'  Random Forest Accuracy : {accuracies[0]:.4f}')
print(f'  Perceptron Accuracy    : {accuracies[1]:.4f}')
print()
print('Random Forest is expected to outperform the Perceptron because:')
print('  - It is a non-linear ensemble model (captures complex patterns)')
print('  - It averages across 200 trees (reduces variance / overfitting)')
print('  - The Perceptron is a linear classifier with no hidden layers')

# ROC Curve for Random Forest (Perceptron has no predict_proba)
fpr, tpr, _ = roc_curve(y_test, rf_proba)
auc = roc_auc_score(y_test, rf_proba)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='steelblue', linewidth=2, label=f'Random Forest (AUC = {auc:.3f})')
plt.plot([0,1],[0,1], 'k--', linewidth=1, label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Random Forest')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
_save()
