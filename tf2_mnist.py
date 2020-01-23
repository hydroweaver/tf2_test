import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

model.save(r'C:\Users\hydro\.spyder-py3\tf2\mnist.h5')

#converter = tf.lite.TFLiteConverter.from_saved_model(r'C:\Users\hydro\.spyder-py3\tf2\mnist.hd5',)
converter = tf.lite.TFLiteConverter.from_keras_model(r'C:\Users\hydro\.spyder-py3\tf2')
tflite_model = converter.convert()

keras_mnist_lite_model_path = r'C:\Users\hydro\.spyder-py3\tf2\mnist.tflite'
open(keras_mnist_lite_model_path, "wb").write(tflite_model)

#From https://www.tensorflow.org/lite/guide/inference#supported_platforms

# Load TFLite model and allocate tensors.
#interpreter = tf.lite.Interpreter(model_path=keras_mnist_lite_model_path)
interpreter = tf.lite.Interpreter(model_path=r'C:\Users\hydro\.spyder-py3\tf2\colab_mnist.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

choice = np.random.choice(np.arange(1, 1000, 1), 10)

random_test_digits = [value for value in list(x_test[choice])]

random_test_digits_labels = list(y_test[choice])

keras_prediction_array = []
tflite_prediction_array = []

for digit in random_test_digits:
    keras_prediction_array.append(model.predict(np.reshape(digit, (1, 28, 28))))

random_test_digits_prediction_keras_model = [np.argmax(prediction) for prediction in keras_prediction_array]

for digit in random_test_digits:
    input_data = np.array(np.reshape(digit, (1, 28, 28)), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    tflite_prediction_array.append(output_data)
    interpreter.reset_all_variables()

random_test_digits_prediction_tflite_model = [np.argmax(prediction) for prediction in tflite_prediction_array]


