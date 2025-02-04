import csv
import random

from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier


# model=Perceptron()
# model=SVC()
model=KNeighborsClassifier(n_neighbors=3)

with open('banknotes.csv') as f:
    reader=csv.reader(f)
    next(reader)
    data=[]
    for row in reader:
        data.append({
            "evidence":[float(cell) for cell in row[:4]],
            "label":"Authentic" if row[4]=="0" else "Counterfeit"
        })



# Separate the test and training data
holdout=int(0.5 * len(data))
random.shuffle(data)
testing=data[:holdout]
training=data[holdout:]


# Training the model
x_training=[row['evidence'] for row in training]
y_training=[row['label'] for row in training]
model.fit(x_training,y_training)

# Test the model
x_test=[row['evidence'] for row in testing]
# Compare the y testing and the 
y_test=[row['label'] for row in testing]
predictions=model.predict(x_test)


# Evaluate the model
correct=0
incorrect=0
total=0

for actual,predicted in zip(y_test,predictions):
    total+=1
    if actual==predicted:
        correct+=1
    else:
        incorrect+=1

print(f"Results from model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {(correct/total)*100:.2f}%")