import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: save figures without a display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load data
df = pd.read_csv("lab4_data.csv")

# 2. Split features/target
X = df.drop(columns=["y"])
y = df["y"]

# handle missing values simply
X = X.fillna(X.mean(numeric_only=True))

# 3. Feature selection - keep top 10 most relevant features
k = min(10, X.shape[1])
selector = SelectKBest(score_func=f_regression, k=k)
X_selected = selector.fit_transform(X, y)
selected_cols = X.columns[selector.get_support()]
print("Selected features:", list(selected_cols))

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# 5. Scale features (important for MLP convergence)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Build MLP model
# 3 hidden layers, decreasing neurons (64-32-16) to gradually compress features
# relu activation: relu(mx+c) style linear combos pass through unchanged when positive,
# giving the network enough non-linearity to model complex regression surfaces
# while avoiding vanishing gradients (unlike sigmoid/tanh)
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

mlp.fit(X_train, y_train)

# 7. Predict and evaluate
y_pred = mlp.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# 8. Diagnostic plots
os.makedirs("plots", exist_ok=True)

# figure_01: training loss curve — shows the network learning over iterations
plt.figure(figsize=(7, 5))
plt.plot(mlp.loss_curve_, color="tab:blue")
plt.title("MLP Training Loss Curve")
plt.xlabel("Iteration (epoch)")
plt.ylabel("Training loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/figure_01.png", dpi=120)
plt.close()

# figure_02: predicted vs actual — points on the diagonal = perfect predictions
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolor="k", linewidth=0.3)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", label="perfect prediction")
plt.title(f"Predicted vs Actual  (R² = {r2:.3f})")
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/figure_02.png", dpi=120)
plt.close()

print("Saved plots to plots/figure_01.png and plots/figure_02.png")