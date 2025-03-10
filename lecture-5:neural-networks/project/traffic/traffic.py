import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images=[]
    labels=[]

    for category in range(NUM_CATEGORIES):
        category_dir=os.path.join(data_dir,str(category))

        if not os.path.isdir(category_dir):
            continue

        for image in os.listdir(category_dir):
            image_path=os.path.join(category_dir,image)
            img=cv2.imread(image_path)

            if image is None:
                continue

            img=cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
            images.append(img)
            labels.append(category)
        
    return images,labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """


    data_augmentation=tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomTranslation(0.1,0.1)
    ])

    model=tf.keras.models.Sequential([

        #Input
        tf.keras.layers.Input(shape=(IMG_WIDTH,IMG_HEIGHT,3)),

        data_augmentation,

        #CNN 1
        tf.keras.layers.Conv2D(32,(3,3),activation="relu",padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        #CNN 2
        tf.keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        # CNN 3
        tf.keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        ## CNN 4
        tf.keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        # Convlution again

        #Flatten
        tf.keras.layers.Flatten(),

        #Hidden Layer 1
        tf.keras.layers.Dense(256,activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        # Hidden Layer 2
        tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        #Output
        tf.keras.layers.Dense(NUM_CATEGORIES,activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Model architecture:")
    model.summary()
    return model


if __name__ == "__main__":
    main()
