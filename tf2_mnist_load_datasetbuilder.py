import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


'''using datasetbuilder - this can be used instead of tfds.load'''

mnist_builder = tfds.builder("mnist")
mnist_builder.download_and_prepare()

mnist_train, mnist_test = mnist_builder.as_dataset(split=['train', 'test'])


'''print a digit'''

example = mnist_train.take(1)

for values in example:
    image, label = values['image'], values['label']

image = image.numpy()
image = np.reshape(image, (28, 28))
print(label.numpy())
plt.imshow(image, cmap=plt.get_cmap('gray'))
plt.show()


print(mnist_builder.info)
print(mnist_builder.info.features)
print(mnist_builder.info.features['label'].num_classes)
print(mnist_builder.info.features['label'].names)
