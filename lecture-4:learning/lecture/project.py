import subprocess
import sys

def ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

ensure("datasets")
ensure("scikit-learn", "sklearn")
ensure("seaborn")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_path(name):
    return os.path.join(PLOTS_DIR, name)

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

sns.set_theme(style="whitegrid")

def load_heart_disease_data():
    try:
        from datasets import load_dataset
        ds = load_dataset("skrishna/heart_disease_uci")
        split = "train" if "train" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()
        if "target" not in df.columns and "num" in df.columns:
            df = df.rename(columns={"num": "target"})
        return df
    except Exception:
        url = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
        return pd.read_csv(url)

def clean_heart_disease_data(df):
    df = df.copy()
    drop_candidates = ["id", "dataset"]
    df = df.drop(columns=[c for c in drop_candidates if c in df.columns])
    if df["target"].nunique() > 2:
        df["target"] = (df["target"] > 0).astype(int)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df

raw_df = load_heart_disease_data()
df = clean_heart_disease_data(raw_df)

target = df["target"]
features = df.drop(columns=["target"])

numeric_cols = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = features.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

plt.figure(figsize=(6, 5))
sns.countplot(x=target, hue=target, palette="viridis", legend=False)
plt.title("Target Class Distribution (0 = No Disease, 1 = Disease)")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(plot_path("target_distribution.png"), dpi=150)

plt.figure(figsize=(11, 9))
corr = pd.concat([features[numeric_cols], target], axis=1).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig(plot_path("correlation_heatmap.png"), dpi=150)

n_hist = min(len(numeric_cols), 9)
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
for ax, col in zip(axes.flatten(), numeric_cols[:n_hist]):
    sns.histplot(features[col], kde=True, ax=ax, color="teal")
    ax.set_title(col)
for ax in axes.flatten()[n_hist:]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(plot_path("feature_distributions.png"), dpi=150)

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

k_best = min(10, len(numeric_cols) + len(categorical_cols))

base_estimators = [
    ("rf", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)),
    ("gb", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)),
    ("svc", SVC(kernel="rbf", probability=True, random_state=42))
]

stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    passthrough=False
)

model_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("feature_selection", SelectKBest(score_func=f_classif, k=k_best)),
    ("classifier", stacking_model)
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model_pipeline, features, target, cv=cv, scoring="accuracy")
metrics["cv_accuracy_mean"] = float(np.mean(cv_scores))
metrics["cv_accuracy_std"] = float(np.std(cv_scores))

print(metrics)
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(plot_path("confusion_matrix.png"), dpi=150)

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {metrics['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(plot_path("roc_curve.png"), dpi=150)

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(recall_vals, precision_vals, color="purple", lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(plot_path("precision_recall_curve.png"), dpi=150)

rf_model = model_pipeline.named_steps["classifier"].named_estimators_["rf"]
selected_mask = model_pipeline.named_steps["feature_selection"].get_support()
ohe_feature_names = []
if categorical_cols:
    ohe = model_pipeline.named_steps["preprocessing"].named_transformers_["cat"].named_steps["encoder"]
    ohe_feature_names = list(ohe.get_feature_names_out(categorical_cols))
all_feature_names = numeric_cols + ohe_feature_names
selected_names = [name for name, keep in zip(all_feature_names, selected_mask) if keep]
importances = rf_model.feature_importances_

plt.figure(figsize=(8, 6))
order = np.argsort(importances)[::-1]
sns.barplot(x=importances[order], y=np.array(selected_names)[order], hue=np.array(selected_names)[order], palette="mako", legend=False)
plt.title("Feature Importance (Random Forest, within Stacking Ensemble)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(plot_path("feature_importance.png"), dpi=150)

plt.figure(figsize=(6, 5))
fold_ids = list(range(1, len(cv_scores) + 1))
sns.barplot(x=fold_ids, y=cv_scores, hue=fold_ids, palette="crest", legend=False)
plt.axhline(np.mean(cv_scores), color="red", linestyle="--", label=f"Mean = {np.mean(cv_scores):.3f}")
plt.title("5-Fold Cross-Validation Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(plot_path("cross_validation_scores.png"), dpi=150)

print(f"\nAll plots saved to: {os.path.abspath(PLOTS_DIR)}/")