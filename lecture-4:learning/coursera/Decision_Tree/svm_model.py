import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"

data=pd.read_csv(url)

print(data.sample(3))

# Data Analysis
labels=data.Class.unique()
size=data['Class'].value_counts().values
print(size)


_,axis=plt.subplots()
axis.pie(size,labels=labels, autopct='%1.3f%%')
axis.set_title("Target variable count")
plt.savefig('target_variable_count.png')



correlation_values=data.corr()['Class'].drop('Class')
correlation_values.plot(kind='barh', figsize=(10,6))

std_scaler=StandardScaler()

# ilioc[row, column]
# Examples:
# ilioc[1,1] ---> row 1 & column 
# ilioc[:,[1,2]] ---> all rows & columns 1 & 2
# ilioc[[1,2],:] ---> rows 1 & 2 & all columns
#ilioc[:,1:30] ---> all rows & columns 1 to 30
data.iloc[:,1:30]=std_scaler.fit_transform(data.iloc[:,1:30])
# print(data.iloc[:,1:30].sample(1))
data_matrix=data.values

evidence=data_matrix[:,1:30]
target=data_matrix[:,30]
evidence=normalize(evidence,norm='l1')



x_train,x_test,y_train,y_test=train_test_split(
    evidence,target,test_size=0.2,random_state=42
)

w_train=compute_sample_weight('bal',y_train)

dt_model=DecisionTreeClassifier(max_depth=4,random_state=35)
dt_model.fit(x_train,y_train,sample_weight=w_train)


svm_model=LinearSVC(class_weight='balanced',random_state=31, loss="hinge", fit_intercept=False)
svm_model.fit(x_train,y_train)


svm_prediction=svm_model.predict(x_test)
dt_prediction=dt_model.predict(x_test)

svm_roc_score=roc_auc_score(y_test,svm_prediction)
dt_roc_score=roc_auc_score(y_test,dt_prediction)

print(f"SVM ROC Score: {svm_roc_score}")
print(f"DT ROC Score: {dt_roc_score}")