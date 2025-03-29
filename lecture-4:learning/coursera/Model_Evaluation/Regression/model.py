from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
import pandas as pd
from scipy.stats import skew
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error ,r2_score



data=fetch_california_housing()
x,y=data.data,data.target



x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)

df=pd.DataFrame(data=x_train,columns=data.feature_names)
df['MedHouseVal']=y_train
print(df.describe())

# Plot the distribution
# plt.hist(1e5*y_train, bins=30, color='lightblue', edgecolor='black')
# plt.title(f'Median House Value Distribution\nSkewness: {skew(y_train):.2f}')
# plt.xlabel('Median House Value')
# plt.ylabel('Frequency')
# plt.savefig('distribution.png')

model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)


mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=root_mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

#So, on average, predicted median house prices are off by $33k.
print("MAE: {:.4f}".format(mae))

print("MSE: {:.4f}".format(mse))
print("RMSE: {:.4f}".format(rmse))
# 80% if the predicted variation ie explained by the model
# ie it means, that the 80% of the dependent variable is explained by the independent variable
# and the remaining 20% of the dependent variable is unexplained
print("R2: {:.4f}".format(r2))


# plt.scatter(y_test,y_pred,alpha=0.5,color="blue")
# plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=2)
# plt.title("Random Forest Regression - Actual vs Predicted")
# plt.xlabel("Actual values")
# plt.ylabel("Predicted values")
# plt.savefig("actual_vs_predicted.png")



importances = model.feature_importances_
print(importances)
indices = np.argsort(importances)[::-1]
print(indices)
features = data.feature_names
print(features)

# Plot feature importances
plt.bar(range(x.shape[1]), importances[indices],  align="center")
plt.xticks(range(x.shape[1]), [features[i] for i in indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances in Random Forest Regression")
plt.savefig('feature_importances.png')


# Plot histogram of the residual errors:
residuals= 1e5*(y_test-y_pred)
# plt.hist(residuals,bins=30,color='lightblue',edgecolor='black')
# plt.title(f"Median House value prediction Residuals")
# plt.xlabel("Median House value Prediction Error ($)")
# plt.ylabel("frequency")
# plt.savefig('residuals.png') 


print("Average error = "+ str(int(np.mean(residuals))))
print("Standart Deviation of error = "+ str(int(np.std(residuals))))


residuals_df=pd.DataFrame({
    'Actual':y_test,
    'Residuals':residuals
})
residuals_df=residuals_df.sort_values(by='Actual')




# plt.scatter(residuals_df['Actual'],residuals_df['Residuals'],marker='o',alpha=0.4,ec='k')
# plt.title('Median House Value Prediciton Residuals Ordered by Actual Median Prices')
# plt.xlabel('Actual Values (Sorted)')
# plt.ylabel('Residuals')
# plt.grid(True)
# plt.savefig('residuals_ordered.png')



