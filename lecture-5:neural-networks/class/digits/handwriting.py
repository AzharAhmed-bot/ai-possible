import sys
import tensorflow as tf


mnist=tf.keras.datasets.mnist


(x_training,y_training),(x_testing,y_testing)=mnist.load_data()
x_training,x_testing=x_training/255.0,x_testing/255.0
y_training=tf.keras.utils.to_categorical(y_training)
y_testing=tf.keras.utils.to_categorical(y_testing)
x_training=x_training.reshape(
    x_training.shape[0],x_training.shape[1],x_training.shape[2],1
)
x_testing=x_testing.reshape(
    x_testing.shape[0],x_testing.shape[1],x_testing.shape[2],1
)

model=tf.keras.models.Sequential([
    
    tf.keras.layers.Input(shape=(28,28,1)),

    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),

    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10,activation="softmax")
])


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_training,y_training,epochs=10)

model.evaluate(x_testing,y_testing,verbose=2)


if len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    filename="model.keras"

model.save(filename)
print(f"Model saved to {filename}.")