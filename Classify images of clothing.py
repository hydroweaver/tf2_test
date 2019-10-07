# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 19:33:39 2019

@author: Karan.Verma
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_train, x_test = x_train / 255, x_test / 255

plt.figure(figsize=(10, 10))
for figure in range(25):
    plt.subplot(5, 5, figure+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[figure], cmap = plt.cm.binary)
    plt.xlabel(class_names[y_train[figure]])
plt.show()


model = keras.models.Sequential([
        keras.layers.Flatten(input_shape = (28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')])
    
model.compile(optimiser = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,
          y_train,
          epochs = 10)


test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)

predictions = model.predict(x_test)

print(np.argmax(predictions[0]))

model.save(r'C:\Users\hydro\.spyder-py3\tf2\fashion_mnist\model.h5')

np.save(r'C:\Users\hydro\.spyder-py3\tf2\predictions', predictions)
np.save(r'C:\Users\hydro\.spyder-py3\tf2\test_images', x_test[:1000])
np.save(r'C:\Users\hydro\.spyder-py3\tf2\test_labels', y_test[:1000])








