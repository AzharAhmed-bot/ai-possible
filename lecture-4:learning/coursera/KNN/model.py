import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
print(df.head())


value_count=df['custcat'].value_counts()
print(value_count)


# Plotting the correlation
corr_matrix=df.corr()
# plt.figure(figsize=(10,8))
# sns.heatmap(data=corr_matrix, annot=True,cmap='coolwarm',fmt='.2f',linewidths=0.5)
# plt.savefig('correlation.png')

ascending_corr=abs(df.corr()['custcat'].drop('custcat').sort_values(ascending=False))
print(ascending_corr)


evidence=df.drop('custcat',axis=1)
evidence=evidence.drop(['region','gender','retire','age','address','marital'],axis=1)
target=df['custcat']

std_scaler=StandardScaler()
evidence_std=std_scaler.fit_transform(evidence)

x_training,x_testing,y_training,y_testing=train_test_split(
    evidence_std,target,test_size=0.2,random_state=32
)


k=3
model=KNeighborsClassifier(n_neighbors=k)
model.fit(x_training,y_training)

prediction=model.predict(x_testing)

print("Accuracy:",accuracy_score(y_testing,prediction))

# Choosing the best k value
k=10
accuracy=np.zeros((k))
std_accuracy=np.zeros((k))

for i in range(1,k+1):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(x_training,y_training)
    prediction=model.predict(x_testing)
    accuracy[i-1]=accuracy_score(y_testing,prediction)
    std_accuracy[i-1]=np.std(prediction==y_testing)/ np.sqrt(prediction.shape[0])

# Accuracy is very low because:
# 1. To many weakly correlated features
# 2. KNN treats all features equally,so it can be sensitive to noise
# 3. KNN performance is measured by how much features provide clear boundaries between classes
# 4. KNN IS NOT A LEARNING MODEL BUT A CLASSIFICATION MODEL
print("The best accuracy was with ", accuracy.max(), "with k=",accuracy.argmax()+1)



# Plotting the accuracy
plt.plot(range(1,k+1),accuracy,'g')
plt.fill_between(range(1,k+1),accuracy - 1 * std_accuracy,accuracy + 1 * std_accuracy, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.savefig("accuracy.png")

