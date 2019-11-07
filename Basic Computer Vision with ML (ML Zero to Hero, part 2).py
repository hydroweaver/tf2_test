import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0


model_tf_style = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
    ])


model_tf_style.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

hist = model_tf_style.fit(train_images, train_labels, epochs = 5, verbose=0)

test_loss, test_acc = model_tf_style.evaluate(test_images, test_labels)

for i in np.random.randint(1000, size=100):
    example_image = np.reshape(train_images[i], (1, 28, 28))
    example_truth_label = train_labels[i]
    example_predicted_label = model_tf_style.predict(example_image)
    print('Picked %i, Predicted %i' % (int(example_truth_label), int(np.argmax(example_predicted_label))))





