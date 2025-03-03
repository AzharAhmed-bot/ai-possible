import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df=pd.read_csv(url)

df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1)
df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',],axis=1)

corrrel=df.corr()
print(corrrel)

axes = pd.plotting.scatter_matrix(df, alpha=0.2)
# need to rotate axis labels so we can read them
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.savefig("relation.png")

evidence=df.iloc[:,[0,1]].to_numpy()
label=df.iloc[:,[2]].to_numpy()


# Pre processing 
# Scaler so that the model doesnt not favor any feature due to it magnitude
# For some reason it has a mean of zero and standard deviation of 1
# I think it does this by refactoring the values such that the mean is 0 and the standard deviation is 1
# Example:
# Consider  x to be values ranging : [1.2, 2.4, 3.6]
# Consider x2 to values : [1500, 1800, 2000]
# Mean is 2.4 ----- Std==0.98
# Mean is 1866.67 ---- std=155.3
# Thus the standard value will be calculated using --- std_value=(value-mean /std)
# This will result in values whos mean is 0 and std is 1
std_scaler=preprocessing.StandardScaler()
evidence_std=std_scaler.fit_transform(evidence)


print(pd.DataFrame(evidence_std).describe().round(2))

x_train,x_test,y_train,y_test=train_test_split(
    evidence_std, label, test_size=0.2,random_state=42
)

model=LinearRegression()
model.fit(x_train,y_train)

coef_=model.coef_
intercept_=model.intercept_
print("Coefficients: ",coef_)
print("Intercept: ",intercept_)

# But since we scalered the data
mean_=std_scaler.mean_
std_devs_=np.sqrt(std_scaler.var_)

coef_orginial=coef_ / std_devs_
intercept_original=intercept_ - np.sum((mean_* coef_)/std_devs_)

print("Original coefficent", coef_orginial)
print("Original intercept", intercept_original)

# ------------------------------------------------------------------
X1=x_test[:,0] if x_test.ndim > 1 else x_test
X2=x_test[:,1] if x_test.ndim > 1 else np.zeros_like(X1)

x1_surf,x2_surf=np.meshgrid(np.linspace(X1.min(),X1.max(),100),
                            np.linspace(X2.min(),X2.max(),100))


y_surf=intercept_+coef_[0,0] * x1_surf + coef_[0,1] * x2_surf


prediction=model.predict(x_test.reshape(-1,1)) if x_test.ndim==1 else model.predict(x_test)

above_plane=y_test >=prediction
below_plane=y_test <prediction
above_plane=above_plane[:,0]
below_plane=below_plane[:,0]


fig=plt.figure(figsize=(20,8))
ax=fig.add_subplot(111,projection='3d')

ax.scatter(X1[above_plane],X2[above_plane], prediction[above_plane], label="Above Plane",s=70,alpha=.7,ec='k')
ax.scatter(X1[below_plane],X2[below_plane], prediction[below_plane], label="Below Plane",s=50,alpha=.3,ec='k')


ax.plot_surface(x1_surf,x2_surf, y_surf,color="k",alpha=0.21,label="plane")

# Set view and labels
ax.view_init(elev=10)

ax.legend(fontsize='x-large',loc='upper center')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(None, zoom=0.75)
ax.set_xlabel('ENGINESIZE', fontsize='xx-large')
ax.set_ylabel('FUELCONSUMPTION', fontsize='xx-large')
ax.set_zlabel('CO2 Emissions', fontsize='xx-large')
ax.set_title('Multiple Linear Regression of CO2 Emissions', fontsize='xx-large')
plt.tight_layout()
plt.savefig("plane.png")
# --------------------------------------------------------------------------------

print(x_train[:,0])

plt.scatter(x_train[:,0],y_train,color="blue")
plt.plot(x_train[:,0], intercept_[0]+ coef_[0,0] * x_train[:,0], "-r")
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.savefig("engine_emission.png")




plt.scater(x_train[:,1],y_train,color="blue")
plt.plot(x_train[:,1],intercept_[0]+ coef_[0,1] * x_train[:,1], "-r")
plt.xlabel("FUELCONSUMPTION")
plt.ylabel("Emission")
plt.savefig("fuel_emission.png")