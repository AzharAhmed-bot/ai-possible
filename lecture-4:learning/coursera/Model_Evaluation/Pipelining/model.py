

# Pipelining stages must each implement a fit and transform methods and the final
# step being the model predicting
# The entire pipeline can be trained simultaneously using GridSearchCV
# Resulting in a self contained predictor model

# Pipelines are essential for scenarios where preprocessing involves estimators performing operations like scaling, encoding categorical variables, imputing missing values, and dimensionality reduction. Pipelines ensure these steps are reproducibly applied to both training and test data.



from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,train_test_split,StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


data=load_iris()
X,y=data.data,data.target
labels=data.target_names


pipeline=Pipeline([
    ('scaler',StandardScaler()),
    ('pca',PCA(n_components=2)),
    ('knn',KNeighborsClassifier(n_neighbors=5))
])

x_train,x_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)
pipeline.fit(x_train,y_train)

test_score=pipeline.score(x_test,y_test)
print(f"Test score: {test_score}")

y_predictions=pipeline.predict(x_test)

conf_matrix=confusion_matrix(y_test,y_predictions)

# plt.figure()
# sns.heatmap(conf_matrix,annot=True,cmap='Blues',fmt='d',xticklabels=labels,yticklabels=labels)
# plt.title('Classification Pipeline confusion Matrix')
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.tight_layout()
# plt.savefig('confusion_matrix_1.png')



# Now lets tune the model using cross validation

pipeline_2=Pipeline([
    ('scaler',StandardScaler()),
    ('pca',PCA()),
    ('knn',KNeighborsClassifier())
])


param_grids={
    "pca__n_components":[2,3],
    "knn__n_neighbors":[3,5,7]
}
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

best_model=GridSearchCV(estimator=pipeline_2,param_grid=param_grids,cv=cv,scoring='accuracy',verbose=2)

best_model.fit(x_train,y_train)
test_score_2=best_model.score(x_test,y_test)
print(f"Test score after tuning: {test_score_2}")
print("Best params", best_model.best_params_)

y_predictions_2=best_model.predict(x_test)
conf_matrix_2=confusion_matrix(y_test,y_predictions_2)

plt.figure()
sns.heatmap(conf_matrix_2,annot=True,cmap='Blues',fmt='d',xticklabels=labels,yticklabels=labels)
plt.title('Classification Pipeline confusion Matrix after tuning')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig('confusion_matrix_2.png')