import csv
import os
import numpy as np
from sklearn.model_selection import train_test_split
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


with open('banknotes.csv') as f:
    reader=csv.reader(f)
    next(reader)
    data=[]
    for row in reader:
        data.append({
            "evidence":[float(cell) for cell in row[:-1]],
            "label":1 if row[-1]=="0" else 0
        })



evidence=np.array([row["evidence"] for row in data],dtype=np.float32)
labels=np.array([row["label"] for row in data],dtype=np.float32)

X_training,X_testing,Y_training,Y_testing=train_test_split(
    evidence,labels,test_size=0.4
)

 
model=tf.keras.models.Sequential()

# Add hidden layer with 8 units
model.add(tf.keras.layers.Dense(8,input_shape=(4,),activation="relu"))

# Add output layer with 1 unit
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_training,Y_training,epochs=20)

model.evaluate(X_testing,Y_testing,verbose=2)