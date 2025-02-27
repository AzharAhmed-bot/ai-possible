
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score

url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url)
df.sample(5)
df.describe()

cdf=df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
cdf.sample(9)


# plt.figure(figsize=(10,6))
# cdf.hist()
# plt.savefig("histogram.png")

# plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS,color='blue')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
# plt.xlim(0,27)
# plt.savefig("regression.png")


evidence=cdf.ENGINESIZE.to_numpy()
label=cdf.CO2EMISSIONS.to_numpy()

X_training,X_testing,Y_training,Y_testing=train_test_split(
    evidence,label,test_size=0.4,random_state=42
)

print(type(X_training)[1])
print(np.shape(X_training))

model=LinearRegression()

model.fit(X_training.reshape(-1,1),Y_training)

print("Coefficients: ",model.coef_[0])
print("Intercept: ",model.intercept_)

# plt.scatter(X_training,Y_training,color="blue")
# plt.plot(X_training,model.coef_* X_training +model.intercept_,"-r")
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.savefig("model.png")


predictions=model.predict(X_testing.reshape(-1,1))

plt.scatter(X_testing,Y_testing,color="red")
plt.plot(X_testing,model.coef_ * X_testing+model.intercept_, "-r")
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.savefig("testing.png")


print("Mean absolute error", mean_absolute_error(Y_testing,predictions))
print("Mean squared error", mean_squared_error(Y_testing,predictions))
print("Root mean squared error", mean_squared_error(Y_testing,predictions))
print("R2 score",r2_score(Y_testing,predictions))
