from sklearn.metrics import explained_variance_score, mean_absolute_error,mean_squared_error,r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split



def regression_result(y_train,y_pred,reg_type):
    ev=explained_variance_score(y_train,y_pred)
    mae=mean_absolute_error(y_train,y_pred)
    r2=r2_score(y_train,y_pred)

    print('Evaluation metrics for '+ reg_type + ' Linear regression')
    print('Explained Variance Score:', round(ev))
    print('Mean Absolute Error:', round(mae))
    print('R2 Score:', round(r2))

noise_factor=1
np.random.seed(42)

# Generate 1000 random numbers with 2D array
x=2*np.random.rand(1000,1)
y=4 + 3 * x + noise_factor*np.random.rand(1000,1)
y_ideal= 4 + 3 * x

print(y.shape)
y_outlier=pd.Series(y.reshape(-1).copy())
print(y_outlier.head())

threshold=1.5
outlier_indices=np.where(x.flatten() > threshold)[0]
print(outlier_indices)

num_outliers=5
selected_indices=np.random.choice(outlier_indices,num_outliers,replace=False)
print(selected_indices)
y_outlier[selected_indices]+=np.random.uniform(50,100,num_outliers)

print(y_outlier.sample())



# Plot the data with outliers and the ideal fit line
# plt.figure(figsize=(12,6))
# plt.scatter(x,y_outlier,ec='k',label="Original Data with outliers")
# plt.plot(x,y_ideal,c='g', label="Ideal, noise free data")
# plt.xlabel("Feature X")
# plt.ylabel("Feature Y")
# plt.legend()
# plt.savefig('outliers.png')


# Plot the data without outliers and the ideal fit line
# plt.figure(figsize=(12,6))
# plt.scatter(x,y,ec='k',label="Data without outliers")
# plt.plot(x,y_ideal,c='g', label="Ideal, noise free data")
# plt.xlabel("Feature X")
# plt.ylabel("Feature Y")
# plt.legend()
# plt.savefig('no_outliers.png')


regular_model=LinearRegression()
regular_model.fit(x,y_outlier)
regular_y_pred=regular_model.predict(x)


ridge_model=Ridge(alpha=1)
ridge_model.fit(x,y_outlier)
ridge_y_pred=ridge_model.predict(x)

lasso_model=Lasso(alpha=.2)
lasso_model.fit(x,y_outlier)
lasso_y_pred=lasso_model.predict(x)

regression_result(y_outlier,regular_y_pred,'Regular')
regression_result(y_outlier,ridge_y_pred,'Ridge')
regression_result(y_outlier,lasso_y_pred,'Lasso')


# plt.figure(figsize=(12,6))
# plt.scatter(x,y,alpha=0.4,ec='k',label="Original Data")
# plt.plot(x,y_ideal,c='g', label="Ideal, noise free data")
# plt.plot(x,regular_y_pred,c='r',linewidth=5, label="Regular")
# plt.plot(x,ridge_y_pred,c='b',linestyle='--', label="Ridge")
# plt.plot(x,lasso_y_pred,c='y',linewidth=2, label="Lasso")
# plt.xlabel("Feature X")
# plt.ylabel("Feature Y")
# plt.legend()
# plt.savefig('regression.png')


# -------------------------------------------------------------------------------------------------------------------------------------------#
from sklearn.datasets import make_regression


x,y,ideal_coef=make_regression(n_samples=100,n_features=100,n_informative=10,noise=10,random_state=42,coef=True)
# Matrix multiplication
ideal_predictions= x @ ideal_coef

x_train,x_test,y_train,y_test,ideal_train,ideal_test=train_test_split(
    x,y,ideal_predictions,test_size=0.2,random_state=42
)

lasso=Lasso(alpha=0.1)
ridge=Ridge(alpha=1.0)
linear=LinearRegression()

lasso.fit(x_train,y_train)
ridge.fit(x_train,y_train)
linear.fit(x_train,y_train)

lasso_predictions=lasso.predict(x_test)
ridge_predictions=ridge.predict(x_test)
linear_predctions=linear.predict(x_test)

print()
print("Second Modelling")
regression_result(y_test,lasso_predictions,'Lasso')
regression_result(y_test,ridge_predictions,'Ridge')
regression_result(y_test,linear_predctions,'Linear')


# Plot predictions vs actual
# fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

# axes[0,0].scatter(y_test, linear_predctions, color="red", label="Linear")
# axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
# axes[0,0].set_title("Linear Regression")
# axes[0,0].set_xlabel("Actual",)
# axes[0,0].set_ylabel("Predicted",)

# axes[0,2].scatter(y_test, lasso_predictions, color="blue", label="Lasso")
# axes[0,2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
# axes[0,2].set_title("Lasso Regression",)
# axes[0,2].set_xlabel("Actual",)

# axes[0,1].scatter(y_test, ridge_predictions, color="green", label="Ridge")
# axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
# axes[0,1].set_title("Ridge Regression",)
# axes[0,1].set_xlabel("Actual",)

# axes[0,2].scatter(y_test, lasso_predictions, color="blue", label="Lasso")
# axes[0,2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
# axes[0,2].set_title("Lasso Regression",)
# axes[0,2].set_xlabel("Actual",)


# # Line plots for predictions compared to actual and ideal predictions
# axes[1,0].plot(y_test, label="Actual", lw=2)
# axes[1,0].plot(linear_predctions, '--', lw=2, color='red', label="Linear")
# axes[1,0].set_title("Linear vs Ideal",)
# axes[1,0].legend()
 
# axes[1,1].plot(y_test, label="Actual", lw=2)
# # axes[1,1].plot(ideal_test, '--', label="Ideal", lw=2, color="purple")
# axes[1,1].plot(ridge_predictions, '--', lw=2, color='green', label="Ridge")
# axes[1,1].set_title("Ridge vs Ideal",)
# axes[1,1].legend()
 
# axes[1,2].plot(y_test, label="Actual", lw=2)
# axes[1,2].plot(lasso_predictions, '--', lw=2, color='blue', label="Lasso")
# axes[1,2].set_title("Lasso vs Ideal",)
# axes[1,2].legend()
 
# plt.tight_layout()
# plt.savefig('actual_vs_prediction.png')



# Compare the model coefficients
linear_coeff=linear.coef_
ridge_coeff=ridge.coef_
lasso_coeff=lasso.coef_

x_axis=np.arange(len(linear_coeff))
x_labels=np.arange(min(x_axis),max(x_axis),10)

# plt.figure(figsize=(12, 6))
# plt.scatter(x_axis, ideal_coef,  label='Ideal', color='blue', ec='k', alpha=0.4)
# plt.bar(x_axis - 0.25, linear_coeff, width=0.25, label='Linear Regression', color='blue')
# plt.bar(x_axis, ridge_coeff, width=0.25, label='Ridge Regression', color='green')
# plt.bar(x_axis + 0.25, lasso_coeff, width=0.25, label='Lasso Regression', color='red')

# plt.xlabel('Feature Index')
# plt.ylabel('Coefficient Value')
# plt.title('Comparison of Model Coefficients')
# plt.xticks(x_labels)
# plt.legend()
# plt.savefig('coefficient_comparison.png')



# plt.figure(figsize=(12, 6))

# plt.bar(x_axis - 0.25, ideal_coef - linear_coeff, width=0.25, label='Linear Regression', color='blue')
# plt.bar(x_axis, ideal_coef - ridge_coeff, width=0.25, label='Ridge Regression', color='green')
# # plt.bar(x_axis + 0.25, ideal_coef - lasso_coeff, width=0.25, label='Lasso Regression', color='red')
# plt.plot(x_axis, ideal_coef - lasso_coeff, label='Lasso Regression', color='red')

# plt.xlabel('Feature Index')
# plt.ylabel('Coefficient Value')
# plt.title('Comparison of Model Coefficient Residuals')
# plt.xticks(x_labels)
# plt.legend()
# plt.savefig('conefficient_residuals.png')



threshold=5

feature_importance_df=pd.DataFrame({
    "lasso Coefficients":lasso_coeff,
    "Ideal Coefficients":ideal_coef
})

feature_importance_df['feature Selected']=feature_importance_df['lasso Coefficients'].abs() > threshold

print("Features Identified as Important by Lasso:")
print(feature_importance_df['feature Selected'])

print("\nNonzero Ideal Coefficient Indices")
print(feature_importance_df['Ideal Coefficients']>0)