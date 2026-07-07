import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("/home/azhar/AIcs50/ML-WORK/data/heating_oil_dataset.csv")

features = df[['Temp (°F)', 'Insulation (in)']]
target = df['Oil (Gal)']

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.1, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print(model.intercept_)
print(model.coef_)


temp_range = np.linspace(df['Temp (°F)'].min(), df['Temp (°F)'].max(), 30)
ins_range = np.linspace(df['Insulation (in)'].min(), df['Insulation (in)'].max(), 30)
temp_grid, ins_grid = np.meshgrid(temp_range, ins_range)

X_grid = pd.DataFrame({
    'Temp (°F)': temp_grid.ravel(),
    'Insulation (in)': ins_grid.ravel()
})
z_grid = model.predict(X_grid).reshape(temp_grid.shape)


# Predict
temp =15
insulation = 10
pred = model.predict([[temp, insulation]])
print(pred)


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Regression plane
ax.plot_surface(temp_grid, ins_grid, z_grid, alpha=0.4, color='tomato', label='Regression plane')

# Actual data points
ax.scatter(df['Temp (°F)'], df['Insulation (in)'], target,
           color='steelblue', edgecolors='black', s=60, zorder=5, label='Data points')

# Labels
ax.set_xlabel("Temperature (°F)")
ax.set_ylabel("Insulation (in)")
ax.set_zlabel("Oil (Gal)")
ax.set_title("3D Regression Plane - Heating Oil Consumption")
ax.legend()

plt.tight_layout()
plt.savefig("heating_oil_3d.png", dpi=150)
plt.show()