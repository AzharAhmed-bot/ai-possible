import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load data.csv — features X1 to X_n, last column is the target
df = pd.read_csv('data.csv')

print('Shape:', df.shape)
print('\nFirst 5 rows:')
print('\nColumn names:')
print(list(df.columns))

# Assume last column is the target (label). Adjust if different.
target_col = df.columns[-1]
print(f'Target column: {target_col}')

X = df.drop(columns=[target_col])
y = df[target_col]

print(f'Features shape : {X.shape}')
print(f'Target shape   : {y.shape}')

# 80/20 train-test split with a fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f'\nTrain size: {X_train.shape}, Test size: {X_test.shape}')

# Scale features — important for Lasso and Ridge to work correctly
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train, transform train
X_test_scaled  = scaler.transform(X_test)        # transform test using same scaler

# Lasso (L1 regularisation) shrinks irrelevant feature coefficients to exactly 0,
# effectively selecting only the most important features.
# alpha controls regularisation strength: higher alpha → more features zeroed out.
lasso = Lasso(alpha=0.01, max_iter=10000, random_state=42)
lasso.fit(X_train_scaled, y_train)

# Extract features with non-zero coefficients
feature_names = X.columns
lasso_coefs   = pd.Series(lasso.coef_, index=feature_names)
selected_features = lasso_coefs[lasso_coefs != 0].index.tolist()

print(f'Total features     : {len(feature_names)}')
print(f'Selected by Lasso  : {len(selected_features)}')
print(f'\nSelected features : {selected_features}')

# Visualise Lasso coefficients — non-zero ones are the selected features
plt.figure(figsize=(14, 5))
lasso_coefs[lasso_coefs != 0].plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Lasso Selected Feature Coefficients (alpha=0.01)', fontsize=13)
plt.xlabel('Feature')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Subset datasets to only the Lasso-selected features
X_train_sel = pd.DataFrame(X_train_scaled, columns=feature_names)[selected_features]
X_test_sel  = pd.DataFrame(X_test_scaled,  columns=feature_names)[selected_features]

# Ridge (L2 regularisation) penalises large coefficients without zeroing them,
# giving a stable model on the selected feature subset.
# alpha=1.0 is a sensible default; tune if needed.
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_sel, y_train)

print('Ridge model fitted on', len(selected_features), 'selected features.')

y_pred_ridge = ridge.predict(X_test_sel)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2   = r2_score(y_test, y_pred_ridge)

print('=== Ridge Regression Performance (Test Set) ===')
print(f'RMSE    : {rmse:.4f}')
print(f'R² Score: {r2:.4f}')
print()
# R² interpretation
if r2 >= 0.9:
    print('Excellent fit (R² ≥ 0.90)')
elif r2 >= 0.7:
    print('Good fit (R² ≥ 0.70)')
else:
    print('Moderate fit — consider tuning alpha or adding features')

# Plot Actual vs Predicted values for the test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_ridge, alpha=0.6, color='steelblue', edgecolors='black', label='Predictions')

# Line of best fit (perfect prediction = diagonal)
min_val = min(y_test.min(), y_pred_ridge.min())
max_val = max(y_test.max(), y_pred_ridge.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.title(f'Ridge Regression — Actual vs Predicted (R²={r2:.3f})', fontsize=13)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
