'''no keras install, using tf.keras'''
'''import data without data builder'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow_datasets as tfds

'''no explicit split'''
mnist = tfds.load("mnist")

'''print a digit'''
example = mnist['train'].take(1)

for values in example:
    image, label = values['image'], values['label']

image = image.numpy().astype(np.float32)
image = np.reshape(image, (28, 28))
print(label.numpy())
plt.imshow(image, cmap=plt.get_cmap('gray'))
plt.show()

'''explicit split'''
mnist_train, mnist_test = tfds.load("mnist", split=['train', 'test'])

'''print a digit'''
example1 = mnist_train.take(1)

for values in example1:
    image, label = values['image'], values['label']

image = image.numpy().astype(np.float32)
image = np.reshape(image, (28, 28))
print(label.numpy())
plt.imshow(image, cmap=plt.get_cmap('gray'))
plt.show()






