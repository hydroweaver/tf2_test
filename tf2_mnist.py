import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255, x_test / 255

model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape = (28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
        ])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,
          y_train,
          epochs = 5)

model.evaluate(x_test, y_test, verbose=2)

keras_mnist_model_path = r'C:\Users\hydro\.spyder-py3\tf2\mnist.hd5'

model.save(keras_mnist_model_path)


converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_mnist_model_path)
tflite_model = converter.convert()

keras_mnist_lite_model_path = r'C:\Users\hydro\.spyder-py3\tf2\mnist.tflite'
open(keras_mnist_lite_model_path, "wb").write(tflite_model)

#From https://www.tensorflow.org/lite/guide/inference#supported_platforms

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=keras_mnist_lite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors.

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)




