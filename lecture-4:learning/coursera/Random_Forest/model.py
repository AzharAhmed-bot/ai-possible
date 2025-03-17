import time
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


data=fetch_california_housing()
evidence=data.data
target=data.target
# Outputs number of rows and number of columns
print(evidence.shape)


x_train,x_test,y_train,y_test=train_test_split(
    evidence,target,test_size=0.2, random_state=35
)

n_estimators=100
rf_model=RandomForestRegressor(n_estimators=n_estimators, random_state=42)
xg_model=XGBRegressor(n_estimators=n_estimators, random_state=42)


# Measure training time of Random Forest
start_time = time.time()
rf_model.fit(x_train, y_train)
end_time = time.time()
print("Training time for Random Forest: {:.2f} seconds".format(end_time - start_time))



# Measure training time of xg boost
start_time = time.time()
xg_model.fit(x_train,y_train)
end_time=time.time()
print("Training time for xgboost: {:.2f} seconds".format(end_time - start_time))


# Random forest 
rf_prediction=rf_model.predict(x_test)
rf_mse=mean_squared_error(y_test,rf_prediction)
rf_r2_score=r2_score(y_test,rf_prediction)
print("Random Forest MSE: {:.2f}".format(rf_mse))
print("Random Forest R2 Score: {:.2f}".format(rf_r2_score))


# XGBoost
xg_prediction=xg_model.predict(x_test)
xg_mse=mean_squared_error(y_test,xg_prediction)
xg_r2_score=r2_score(y_test,xg_prediction)
print("XGBoost MSE: {:.2f}".format(xg_mse))
print("XGBoost R2 Score: {:.2f}".format(xg_r2_score))

std_y=np.std(y_test)


plt.figure(figsize=(15,5))

# Random Forest plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, rf_prediction, alpha=0.5, color="blue",ec='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2,label="perfect model")
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1, )
plt.ylim(0,6)
plt.title("Random Forest Predictions vs Actual")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()



# XGBoost plot
plt.subplot(1,2,2)
plt.scatter(y_test,xg_prediction,alpha=0.5,color="orange",ec='k')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=2,label="perfect model")
plt.plot([y_test.min(),y_test.max()],[y_test.min()+std_y,y_test.max()+std_y],'r--',lw=1,label="+/-1 Std Dev")
plt.plot([y_test.min(),y_test.max()],[y_test.min()-std_y,y_test.max()-std_y],'r--',lw=1)
plt.ylim(0,6)
plt.title("XGBoost Predictions vs Actual")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()

plt.savefig('comparison.png')