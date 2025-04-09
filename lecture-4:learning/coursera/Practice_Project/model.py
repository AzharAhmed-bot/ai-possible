import seaborn as sns
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd


titanic=sns.load_dataset('titanic')
print(titanic.count())

features=['pclass','sex','age','sibsp','parch','fare','class','who','adult_male','alone']
target='survived'

X=titanic[features]
y=titanic[target]

print(y.value_counts())

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42, stratify=y
)

numerical_features=X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features=X_train.select_dtypes(include=['object','category']).columns.tolist()


numerical_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')), # Replace missing values with the data median
    ('scaler',StandardScaler())
])
categorical_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')), # Replace missing values with the most frequent value
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

preprocessor=ColumnTransformer(
    transformers=[
        ('num',numerical_transformer,numerical_features),
        ('cat',categorical_transformer,categorical_features)
    ]
)

pipeline=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',RandomForestClassifier(random_state=42))
])
pipeline_2=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',LogisticRegression(random_state=42))
])


param_grids={
    'classifier__n_estimators':[50,100],
    'classifier__max_depth':[None,10,20],
    'classifier__min_samples_split':[2,5]
}
param_grids_2={
    'classifier__solver':['liblinear','saga'],
    'classifier__penalty':['l1','l2'],
    'classifier__class_weight':[None,'balanced'],
}

cv=StratifiedKFold(n_splits=5,shuffle=True)

model=GridSearchCV(
    estimator=pipeline,
    param_grid=param_grids,
    cv=cv,
    scoring='accuracy',
    verbose=2,
)
model.fit(X_train,y_train)

# Model 2
model_2=GridSearchCV(
    estimator=pipeline_2,
    param_grid=param_grids_2,
    cv=cv,
    scoring='accuracy',
    verbose=2
)
model_2.fit(X_train,y_train)

y_pred=model.predict(X_test)
print(classification_report(y_test,y_pred))
y_pred_2=model_2.predict(X_test)
print(classification_report(y_test,y_pred_2))

conf_matrix=confusion_matrix(y_test,y_pred)
conf_matrix_2=confusion_matrix(y_test,y_pred_2)


sns.heatmap(conf_matrix,annot=True,cmap='Blues',fmt='d')
plt.title("Confusion matrix for Titanic dataset")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig('confusion_matrix.png')

sns.heatmap(conf_matrix_2,annot=True,cmap='Blues',fmt='d')
plt.title("Confusion matrix for Titanic dataset using Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig('confusion_matrix_2.png')


# Trace back to access one hot encoder feature names
feature_names_access=model.best_estimator_['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
print(feature_names_access)


feature_importance=model.best_estimator_['classifier'].feature_importances_
coefficients=model_2.best_estimator_['classifier'].coef_[0]


feature_names=numerical_features + list(feature_names_access)


importance_df=pd.DataFrame({
    "Feature":feature_names,
    "Importance":feature_importance
}).sort_values(by='Importance',ascending=False)
importance_df_2=pd.DataFrame({
    "Feature":feature_names,
    "Coefficient":coefficients
}).sort_values(by="Coefficient",ascending=False,key=abs)




# Plotting
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis() 
plt.title('Most Important Features in predicting whether a passenger survived')
plt.xlabel('Importance Score')
plt.savefig('feature_importance.png')

#Plotting 2
plt.figure(figsize=(10, 6))
plt.barh(importance_df_2['Feature'], importance_df_2['Coefficient'], color='skyblue')
plt.gca().invert_yaxis()
plt.title('Feature Coefficient magnitudes for Logistic Regression model')
plt.xlabel('Coefficient Magnitude')
plt.savefig('feature_importance_2.png')

