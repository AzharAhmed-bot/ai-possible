import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt

path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv"
data=pd.read_csv(path)


label_encoder=LabelEncoder()
categorical_columns=data.select_dtypes(include=['object']).columns.tolist()

for column in categorical_columns:
    data[column]=label_encoder.fit_transform(data[column])

print(data.sample(5))
# print(data.isnull().sum())
corr=data.drop('Drug',axis=1).corr()
print(corr)


evidence=data.drop('Drug',axis=1)
label=data['Drug']


x_train,x_test,y_train,y_test=train_test_split(
    evidence,label,test_size=0.2,random_state=42
)


model=DecisionTreeClassifier()
model.fit(x_train,y_train)
predictions=model.predict(x_test)
plot_tree(model)
plt.savefig("decision_tree.png")

print(f"Accuracy {100* round(accuracy_score(y_test,predictions),2)}%")
