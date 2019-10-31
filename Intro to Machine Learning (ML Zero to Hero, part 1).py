import tensorflow as tf
import numpy as np
from tensorflow import keras
from random import randint

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer = 'sgd',
               loss = 'mean_squared_error',
              metrics=['accuracy'])

x_input = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y_output = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

hist = model.fit(x_input,
                 y_output,
                 epochs=500)

floats  =  [float(x)  for x in range (1, 11)]

for i in floats:
    print('Prediction for %f is %f' % (i, model.predict([i])))




