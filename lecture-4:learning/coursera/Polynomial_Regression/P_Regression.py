import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"

churn_df=pd.read_csv(url)

churn_df=churn_df[['tenure','age','address','income','ed','employ','equip','churn']]

churn_df['churn']=churn_df['churn'].astype('int')


evidence=np.asarray(churn_df[['tenure','age','address','income','ed','employ','equip']])
target=np.asarray(churn_df[['churn']])

evidence_norm=StandardScaler().fit(evidence).transform(evidence)

X_train,X_test,Y_train,Y_test=train_test_split(
    evidence,target,test_size=0.2, random_state=42
)

model=LogisticRegression()
model.fit(X_train,Y_train)

prediction=model.predict(X_test)
prediction_proba=model.predict_proba(X_test)

coefficients=pd.Series(model.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.savefig('logistic_Regression.png')


loss=log_loss(Y_test,prediction_proba)
print("Log Loss: ",loss) 