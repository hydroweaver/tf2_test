import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

plt.clf()
plt.subplots(2, 10)
for i in range(10):
    plt.subplot(10, 2, i+1)
    plt.imshow(train_images[i])
    plt.subplot(10, 2, i+2)
    plt.imshow(test_images[i])
plt.show()

x = np.load(r'C:\Users\hydro\.spyder-py3\tf2\predictions.npy')

print(np.shape(x))
