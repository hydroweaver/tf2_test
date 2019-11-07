#https://codelabs.developers.google.com/codelabs/tensorflow-lab3-convolutions/#4

#used numpy for multiplying instead of manually done in website

import cv2
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

image = misc.ascent()

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(image)
#plt.show()

image_copy = np.copy(image)
image_copy_x = image_copy.shape[0]
image_copy_y = image_copy.shape[1]

#create filter
'''
filter1 = np.array([[0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0]]) # brightens up the pic
filter1 = np.array([[1, 0, 0],
                    [0, -2, 0],
                    [0, 0, 1]]) #curve left
filter1 = np.array([[0, 0, 1],
                    [0, -2, 0],
                    [1, 0, 0]]) #curve right
filter1 = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [0, 0, 0]])
filter1 = np.array([[0, 0, 0],
                    [1, 1, 1],
                    [0, 0, 0]])'''

'''filter1 = np.array([[1, -1, 1, -1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [-1, 1, -1, 1, -1]])''' #applies hazier or more blurry operations on the image for horizontal lines

'''filter1 = np.array([[1, 0, 0, 0, -1],
                    [1, 0, 0, 0, -1],
                    [1, 0, 0, 0, -1],
                    [1, 0, 0, 0, -1],
                    [1, 0, 0, 0, -1]])''' #applies hazier or more blurry operations on the image for vertical lines


weight = 1

plt.ion()
plt.subplots(1, 2)
plt.subplot(1, 2, 1)
plt.imshow(image)


#FOR 3X3 FILTER
for i in range(1, image_copy_x-2):
    for j in range(1, image_copy_y-2):
        if np.sum(filter1*image_copy[i:i+3, j:j+3]) < 0:
            image_copy[i, j] = 0
        elif np.sum(filter1*image_copy[i:i+3, j:j+3]) > 255:
            image_copy[i, j] = 255
        else:
            image_copy[i, j] = np.sum(filter1*image_copy[i:i+3, j:j+3])*weight


#FOR 5X5 FILTER

for i in range(1, image_copy_x-4):
    for j in range(1, image_copy_y-4):
        if np.sum(filter1*image_copy[i:i+5, j:j+5]) < 0:
            image_copy[i, j] = 0
        elif np.sum(filter1*image_copy[i:i+5, j:j+5]) > 255:
            image_copy[i, j] = 255
        else:
            image_copy[i, j] = np.sum(filter1*image_copy[i:i+5, j:j+5])*weight
    
plt.subplot(1, 2, 2)
plt.imshow(image_copy)
plt.show()
