import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

fashion_cnn_data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_cnn_data.load_data()

train_images_reshaped = np.expand_dims(train_images, -1)
test_images_reshaped = np.expand_dims(test_images, -1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
    ])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(train_images_reshaped,
                 train_labels,
                 epochs = 5,
                 verbose=0)

test_loss, test_acc = model.evaluate(test_images_reshaped, test_labels)

for i in np.random.randint(1000, size=100):
    example_image = np.expand_dims(train_images_reshaped[i], 0)
    example_truth_label = train_labels[i]
    example_predicted_label = model.predict(example_image.astype('float32'))
    print('Picked %i, Predicted %i' % (int(example_truth_label), int(np.argmax(example_predicted_label))))
