from sklearn.metrics import explained_variance_score, mean_absolute_error,mean_squared_error,r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge,Lasso



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


plt.figure(figsize=(12,6))
plt.scatter(x,y,alpha=0.4,ec='k',label="Original Data")
plt.plot(x,y_ideal,c='g', label="Ideal, noise free data")
plt.plot(x,regular_y_pred,c='r',linewidth=5, label="Regular")
plt.plot(x,ridge_y_pred,c='b',linestyle='--', label="Ridge")
plt.plot(x,lasso_y_pred,c='y',linewidth=2, label="Lasso")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.legend()
plt.savefig('regression.png')