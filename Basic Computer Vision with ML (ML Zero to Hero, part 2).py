import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


model_tf_style = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
    ])


model_tf_style1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
    ])


'''model_keras_style = keras.models.Sequential()
model_keras_style.add  layers.add(Flatten(input_shape=(28, 28)))
model_keras_style.layers.add(Dense(128, activation = 'relu')
model_keras_style.layers.add(Dense(10, activation = 'softmax')'''


model_tf_style.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy')

hist = model_tf_style.fit(train_images, train_labels, epochs = 5)

test_loss, test_acc = model_tf_style.evaluate(test_images, test_labels)

#predictions = model.predict(


