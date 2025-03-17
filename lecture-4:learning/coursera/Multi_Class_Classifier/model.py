import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier
from sklearn.metrics import accuracy_score

url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data=pd.read_csv(url)

# sns.countplot(y="NObeyesdad",data=data)
# plt.title("Distribution of obesity levels")
# plt.savefig("obesity.png")
# print(data.describe())
# print(data.info())
# print(data.isnull().sum())

# Get neumerical features
continuous_columns=data.select_dtypes(include=['float64']).columns.tolist()

# Standardize the them
scaler=StandardScaler()
scaled_features=scaler.fit_transform(data[continuous_columns])
scaled_df=pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
scaled_data=pd.concat([data.drop(columns=continuous_columns),scaled_df],axis=1)


categorical_columns=data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')
encoder=OneHotEncoder(sparse_output=False, drop='first')
encoded_features=encoder.fit_transform(data[categorical_columns])
print(encoded_features)
print(categorical_columns)
encoded_df=pd.DataFrame(encoded_features,columns=encoder.get_feature_names_out(categorical_columns))


prepepped_data=pd.concat([scaled_data.drop(columns=categorical_columns),encoded_df],axis=1)



# Encode the label/target variable
prepepped_data['NObeyesdad']=prepepped_data['NObeyesdad'].astype('category').cat.codes
# print(prepepped_data.head())

evidence=prepepped_data.drop('NObeyesdad',axis=1)
label=prepepped_data['NObeyesdad']


x_train,x_test,y_train,y_test=train_test_split(
    evidence,label,test_size=0.2,random_state=42
)

ovr_model=LogisticRegression(multi_class="ovr",max_iter=1000)
ovr_model.fit(x_train,y_train)
ovr_prediction=ovr_model.predict(x_test)
print("One vs All Strategy")
print(f"Accuracy: {np.round(accuracy_score(y_test,ovr_prediction),2)*100}%")

coef_=ovr_model.coef_

feature_imporance=np.mean(np.abs(coef_),axis=0)
plt.barh(evidence.columns,feature_imporance)
plt.xlabel("Feature importance")
plt.savefig("ovr_feature.png")




ovo_model=OneVsOneClassifier(LogisticRegression(max_iter=1000))
ovo_model.fit(x_train,y_train)
ovo_prediction=ovo_model.predict(x_test)
print("One vs One Strategy")
print(f"Accuracy: {np.round(100* accuracy_score(y_test,ovo_prediction),2)}%")

coef_=np.array([est.coef_.flatten for est in ovo_model.estimators_])

feature_imporance=np.mean(np.abs(coef_),axis=0)
plt.barh(evidence.columns,feature_imporance)
plt.x_label("Feature importance")
plt.savefig("ovo_feature.png")