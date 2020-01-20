import scipy.misc
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

original_image_path = r'C:\Users\hydro\Downloads\flutter\camera\images\predict.jpg'
greyscale_image_path = r'C:\Users\hydro\Downloads\flutter\camera\images\grey.png' 

greyscale_image = Image.open(original_image_path).convert('LA').rotate(-90)
greyscale_image.save(greyscale_image_path)

imported_image = Image.open(greyscale_image_path)

x = np.asarray(imported_image)

x = plt.imshow(x[:, :, 0], cmap="gray")

plt.show()

z = x[:, :, 0]

y = plt.imsave(r'C:\Users\hydro\Downloads\flutter\camera\images\xx.png', z, cmap="gray")




