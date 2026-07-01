#!/usr/bin/env python
# coding: utf-8

# # Topic 5 — Regression Models
# **Q1:** Multiple Linear Regression — Heating Oil
# 
# **Q2:** Logistic Regression — Golf Putts

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ---
# ## Question 1 — Multiple Linear Regression: Heating Oil Consumption
# **Goal:** Predict heating oil used (gallons) from average temperature (°F) and insulation thickness (inches).

# In[2]:


# Dataset from the question slide
data = {
    'Oil_Gal':     [275.30, 363.80, 164.30,  40.80,  94.30, 230.90,
                    366.70, 300.60, 237.80, 121.40,  31.40, 203.50,
                    441.10, 323.00,  52.50],
    'Temp':        [40,     27,     40,     73,     64,     34,
                     9,      8,     23,     63,     65,     41,
                    21,     38,     58],
    'Insulation':  [ 3,      3,     10,      6,      6,      6,
                     6,     10,     10,      3,     10,      6,
                     3,      3,     10]
}
df = pd.DataFrame(data)
print('Dataset shape:', df.shape)
df.head()


# In[3]:


# --- Part a) Preliminary conclusions from the data ---
print('=== Preliminary Conclusions ===')
print(df.describe())
print()
print('Correlation matrix:')
print(df.corr().round(3))


# ### Part a) — Preliminary Conclusions
# - **Temperature vs Oil:** Strong negative correlation — as outside temperature rises, less oil is consumed (makes intuitive sense: warmer days need less heating).
# - **Insulation vs Oil:** Negative correlation — better insulation (more inches) reduces heating oil consumption.
# - The data confirms that **both predictors are relevant** for estimating oil usage.

# In[4]:


# Visualise the relationships
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(df['Temp'], df['Oil_Gal'], color='steelblue', edgecolors='black')
axes[0].set_xlabel('Average Temperature (°F)')
axes[0].set_ylabel('Oil Consumption (Gal)')
axes[0].set_title('Oil vs Temperature')
axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].scatter(df['Insulation'], df['Oil_Gal'], color='coral', edgecolors='black')
axes[1].set_xlabel('Insulation (inches)')
axes[1].set_ylabel('Oil Consumption (Gal)')
axes[1].set_title('Oil vs Insulation')
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


# In[5]:


# --- Build the Multiple Linear Regression model ---
X = df[['Temp', 'Insulation']]  # two predictors
y = df['Oil_Gal']               # target

model = LinearRegression()
model.fit(X, y)

print('=== Multiple Linear Regression Model ===')
print(f'Intercept     : {model.intercept_:.4f}')
print(f'Coef (Temp)   : {model.coef_[0]:.4f}')
print(f'Coef (Insul.) : {model.coef_[1]:.4f}')
print()
print(f'Regression equation:')
print(f'Oil = {model.intercept_:.2f} + ({model.coef_[0]:.2f} × Temp) + ({model.coef_[1]:.2f} × Insulation)')

y_pred = model.predict(X)
print(f'\nR² Score : {r2_score(y, y_pred):.4f}')
print(f'RMSE     : {np.sqrt(mean_squared_error(y, y_pred)):.4f}')


# In[6]:


# --- Part b) Prediction: Temp = 15°F, Insulation = 10 inches ---
temp_new        = 15
insulation_new  = 10
prediction      = model.predict([[temp_new, insulation_new]])[0]

print(f'=== Part b) Prediction ===')
print(f'Input  : Temperature = {temp_new}°F, Insulation = {insulation_new} inches')
print(f'Predicted Oil Consumption: {prediction:.2f} gallons')


# ---
# ## Question 2 — Logistic Regression: Golf Putts Success Rate
# **Goal:** Predict the proportion of putts made (success) based on putt length (feet).

# In[7]:


# Dataset from the question slide
# Expand grouped data: each putt becomes one row (Made=1, Missed=0)
lengths = [3, 4, 5, 6, 7]
made    = [84, 88, 61, 61, 44]
missed  = [17, 31, 47, 64, 90]

rows = []
for length, m, mi in zip(lengths, made, missed):
    rows += [(length, 1)] * m   # made putts → success = 1
    rows += [(length, 0)] * mi  # missed putts → success = 0

df2 = pd.DataFrame(rows, columns=['Length', 'Success'])
print('Dataset shape:', df2.shape)
print('Success distribution:')
print(df2['Success'].value_counts())
df2.head()


# In[8]:


# --- Observed success proportion per length ---
proportions = [m / (m + mi) for m, mi in zip(made, missed)]
print('Observed Success Proportions by Length:')
for l, p in zip(lengths, proportions):
    print(f'  {l} ft : {p:.3f} ({p*100:.1f}%)')


# In[9]:


# --- Fit Logistic Regression ---
X2 = df2[['Length']]
y2 = df2['Success']

log_model = LogisticRegression()
log_model.fit(X2, y2)

print('=== Logistic Regression Model ===')
print(f'Intercept : {log_model.intercept_[0]:.4f}')
print(f'Coef      : {log_model.coef_[0][0]:.4f}  (for Length)')
print()
print('Logit equation: log(p/(1-p)) =',
      f'{log_model.intercept_[0]:.4f} + {log_model.coef_[0][0]:.4f} × Length')


# In[10]:


# --- Plot: Observed proportions vs Logistic curve ---
x_range  = np.linspace(2, 8, 200).reshape(-1, 1)
prob_pred = log_model.predict_proba(x_range)[:, 1]  # probability of success

plt.figure(figsize=(10, 5))
plt.scatter(lengths, proportions, color='red', s=100, zorder=5, label='Observed proportion')
plt.plot(x_range, prob_pred, color='steelblue', linewidth=2.5, label='Logistic curve')
plt.xlabel('Putt Length (feet)', fontsize=12)
plt.ylabel('P(Made)', fontsize=12)
plt.title('Logistic Regression: Probability of Making a Putt', fontsize=13)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[11]:


# --- Model Evaluation ---
y_pred2 = log_model.predict(X2)
print('=== Model Evaluation ===')
print(classification_report(y2, y_pred2))

# Predict probabilities for each putt length
print('\nPredicted success probability per length:')
for l in lengths:
    p = log_model.predict_proba([[l]])[0][1]
    print(f'  {l} ft → {p:.3f} ({p*100:.1f}%)')

