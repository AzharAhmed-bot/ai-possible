import csv
import random

from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

model=Perceptron()

with open('banknotes.csv') as f:
    reader=csv.reader(f)
    next(reader)
    data=[]
    for row in reader:
        data.append({
            "evidence":[float(cell) for cell in row[:4]],
            "label":"Authentic" if row[4]=="0" else "Counterfeit"
        })
evidence=[row["evidence"] for row in data]
labels=[row["label"] for row in data]

# Train the model
x_training,x_testing,y_training,y_testing=train_test_split(
    evidence,labels,test_size=0.5
)
model.fit(x_training,y_training)

# Test the model
predictions=model.predict(x_testing)

# Evaluate the model
correct=(y_testing==predictions).sum()
incorrect=(y_testing!=predictions).sum()
accuracy=(correct/len(y_testing))*100

print(f"Result for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {accuracy}")
