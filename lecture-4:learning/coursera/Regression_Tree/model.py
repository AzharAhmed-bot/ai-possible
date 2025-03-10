import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error


path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv"
data=pd.read_csv(path)


correlation=data.corr()["tip_amount"].drop('tip_amount')
# print(correlation)
top_three=correlation.sort_values(ascending=False)[:3]
print(top_three)

evidence=data.drop('tip_amount',axis=1)
evidence.drop(['payment_type', 'VendorID', 'store_and_fwd_flag', 'improvement_surcharge'],axis=1)
evidence=evidence.values

target=data[['tip_amount']].values.astype('float32')
# print(target)

evidence=normalize(evidence,norm='l1',copy=False)

x_train,x_test,y_train,y_test=train_test_split(
    evidence, target, test_size=0.2, random_state=32
)

model=DecisionTreeRegressor(criterion='squared_error',max_depth=4,random_state=35)
model.fit(x_train,y_train)
prediction=model.predict(x_test)

mse=mean_squared_error(y_test,prediction)
r2=r2_score(y_test,prediction)

print('MSE score : {0:.3f}'.format(mse))
print('R2 score : {0:.3f}'.format(r2))