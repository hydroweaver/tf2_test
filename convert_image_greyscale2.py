import scipy.misc
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

original_image_path = r'C:\Users\hydro\Downloads\flutter\camera\images\predict.jpg'
greyscale_image_path = r'C:\Users\hydro\Downloads\flutter\camera\images\grey111.png'


tflite_model_path = r'C:\Users\hydro\Downloads\flutter\camera\model\mnist.tflite'

gray_image = Image.open(original_image_path, mode="r").rotate(-90)
gray_image11 = Image.open(greyscale_image_path, mode="r").rotate(-90)
gray_image23 = plt.imread(greyscale_image_path)

gray_image = np.asarray(gray_image, dtype=np.uint8()) / 255.0

#plt.imshow(x[:, :, 0], cmap="gray")
#plt.imsave(greyscale_image_path,gray_image[:, :, 0], cmap='gray')

#y = plt.imread(greyscale_image_path)

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

reshape_input = np.reshape(gray_image[:, :, 0], (1, 28, 28))

input_data = np.array(reshape_input, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)





