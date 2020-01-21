import io
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


file_path = r'C:\Users\Karan.Verma\Downloads\flutter\camera\images\grey111.png'

image = Image.open(file_path, mode='r')

x = np.asarray(image)

arr = x.copy()


# arr[:, :, 0] shape is (28, 28)

for i in range(0, len(arr[:, :, 0])):
    for j in range(0, len(arr[:, :, 0])):
        arr[i, j, 0] = 1 - arr[i, j, 0]/255

plt.subplots(1, 4)
plt.subplot(141)
plt.imshow(arr[:, :, 0])
plt.subplot(142)
plt.imshow(arr[:, :, 1])
plt.subplot(143)
plt.imshow(arr[:, :, 2])
plt.subplot(144)
plt.imshow(arr[:, :, 3])
plt.show()




