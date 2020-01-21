import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter


original_image_path = r'C:\Users\Karan.Verma\Downloads\flutter\camera\images\predict.jpg'

im = Image.open(original_image_path).convert('LA').rotate(-90)
im_data = np.asarray(im)
im2 = im.resize((28, 28), Image.ANTIALIAS)
im3 = im2.filter(ImageFilter.SHARPEN)
im4 = im.filter(ImageFilter.SHARPEN)
im5 = im.filter(ImageFilter.EDGE_ENHANCE)

#Matplotlib can only read PNGs natively. Further image formats are supported via the optional dependency on Pillow. Note, URL strings are not compatible with Pillow. Check the Pillow documentation for more information.
#imread doesnt work with non png images, so we use Image lib, and hence the error : TypeError: Object does not appear to be a 8-bit string path or a Python file-like object

#newImg = Image.new(mode='L', size=(28, 28), color=(255))

#REMEMBER, NO MATTER HOW THE IMAGE IS, 28X28, OR 28X28X4, WHEN YOU LOAD USING IMSHOW, IT ADDS CHANNELS TO IMAGE TO SHOW IT, AND SO YOU DONT SEE A 28, 28 IMAGE, BECAUSE IT CANT BE LOADED PERHAPS, LIKE TRYING TO SEE A 4D IMAGE ! YOU CANT SEE IT !

plt.subplots(1, 4)
plt.subplot(141)
plt.imshow(im)
plt.subplot(142)
plt.imshow(im2)
plt.subplot(143)
plt.imshow(im3)
plt.subplot(144)
plt.imshow(im5)
plt.show()


