from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


data=load_breast_cancer()
evidence,target=data.data, data.target
labels=data.target_names
feature_names=data.feature_names


# Scale the data
scaler=StandardScaler()
evidence_scaled=scaler.fit_transform(evidence)




# Add Gaussain noise
np.random.seed(42)
nosie_factor=0.5
evidence_noisy=evidence_scaled + nosie_factor * np.random.normal(loc=0.0,scale=1.0,size=evidence.shape)


df=pd.DataFrame(evidence_scaled,columns=feature_names)
df_noisy=pd.DataFrame(evidence_noisy,columns=feature_names)

# print("Original Data:")
# print(df.head())
# print("\nNoisy Data:")
# print(df_noisy.head()) 


# print(df[feature_names])


plt.figure(figsize=(12,6))

# Plotting original distribution
plt.subplot(1,2,1)
plt.hist(df[feature_names[5]],bins=50,alpha=0.7, color='b',label="Original")
plt.title("Original Feature distribution")
plt.xlabel(feature_names[5])
plt.ylabel("Frequency")

# Plotting noisy distribution
plt.subplot(1,2,2)
plt.hist(df_noisy[feature_names[5]],bins=50,alpha=0.7,color='red',label="Noisy")
plt.title("Noisy Feature distribution")
plt.xlabel(feature_names[5])
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig('distribution.png')


# Plotting both original and noisy data comparison
plt.figure(figsize=(12,6))
plt.plot(df[feature_names[5]],label='Original',lw=3)
plt.plot(df_noisy[feature_names[5]],'--',label='Noisy')
plt.title("Scaled feature comparison with and without noise")
plt.xlabel(feature_names[5])
plt.legend()
plt.tight_layout()
plt.savefig('comparison.png')

# Plotting both original and noisy data comparison
plt.figure(figsize=(12, 6))
plt.scatter(df[feature_names[5]], df_noisy[feature_names[5]],lw=5)
plt.title('Scaled feature comparison with and without noise')
plt.xlabel('Original Feature')
plt.ylabel('Noisy Feature')
plt.tight_layout()
plt.savefig('scatter.png')


x_train,x_test,y_train,y_test=train_test_split(
    evidence_scaled,target,test_size=0.2,random_state=42
)


knn_model=KNeighborsClassifier(n_neighbors=5)
svm_model=SVC(kernel='linear',C=1,random_state=42)

knn_model.fit(x_train,y_train)
svm_model.fit(x_train,y_train)


knn_prediction=knn_model.predict(x_test)
svm_prediction=svm_model.predict(x_test)



knn_accuracy_score=accuracy_score(y_test,knn_prediction)
svm_accuracy_score=accuracy_score(y_test,svm_prediction)
knn_classification_report=classification_report(y_test,knn_prediction)
svm_classification_report=classification_report(y_test,svm_prediction)


print(f"KNN Accuracy Score: {knn_accuracy_score}")
print(f"SVM Accuracy Score: {svm_accuracy_score}")
print(f"KNN Classification Report:\n {knn_classification_report}")
print(f"SVM Classification Report:\n {svm_classification_report}")


conf_matrix_knn = confusion_matrix(y_test, knn_prediction)
conf_matrix_svm = confusion_matrix(y_test, svm_prediction)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(conf_matrix_knn, annot=True, cmap='Blues', fmt='d', ax=axes[0],
            xticklabels=labels, yticklabels=labels)

axes[0].set_title('KNN Testing Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(conf_matrix_svm, annot=True, cmap='Blues', fmt='d', ax=axes[1],
            xticklabels=labels, yticklabels=labels)
axes[1].set_title('SVM Testing Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrix.png')


# NOTES:
# SVM outperformed KNN in terms of precision, recall, 
# and F1-score for both for the individual classes and their overall averages. This indicates that SVM is a stronger classifier. 
# Although KNN performed quite well with an accuracy of 94%, 
# SVM has better ability to correctly classify both malignant and beinign cases, with fewer errors. 
# Given that the goal would be to choose the model with better generalization and fewer false negatives, SVM is certainly the preferred classifier.