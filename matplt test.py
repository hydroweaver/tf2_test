import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

main_dir = 

'''train_images = np.load(r'C:\Users\hydro\.spyder-py3\tf2\train_images.npy')
train_labels = np.load(r'C:\Users\hydro\.spyder-py3\tf2\train_labels.npy')
predictions = np.load(r'C:\Users\hydro\.spyder-py3\tf2\predictions.npy')'''



train_images = np.load(r\train_images.npy')
train_labels = np.load(r'C:\Users\hydro\.spyder-py3\tf2\train_labels.npy')
predictions = np.load(r'C:\Users\hydro\.spyder-py3\tf2\predictions.npy')


#train_labels = tf.one_hot(train_labels, 10)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.subplots(2, 2)


def plot_image(i, class_names, train_labels, predictions):
    class_name, train_label_text, train_label, prediction = class_names[i], class_names[train_labels[i]], train_labels[i], predictions[i]
    plt.subplot(2, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.xlabel('Original {} & Detected {} with {:2.0f}% accuracy'.\
               format(class_names[train_labels[i]], class_names[np.argmax(predictions[i])],\
                      100*np.max(predictions[i])))
    plt.imshow(train_images[i])

def plot_barch(i, predictions):
    plt.subplot(2, 2, 2)
    plt.bar(range(10), predictions[i])

                                                            
i = np.random.randint(0, 10)
plot_image(i, class_names, train_labels, predictions)
plot_barch(i, predictions)

#plt.bar(range(10), predictions[i])[np.argmax(predictions[i]r)].set_color('blue')
#plt.bar(range(10), predictions[i])[train_labels[i]].set_color('red')

i = np.random.randint(0, 10)
plt.subplot(2, 2, 3)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.xlabel('Original {} & Detected {} with {:2.0f}% accuracy'.\
               format(class_names[train_labels[i]], class_names[np.argmax(predictions[i])],\
                      100*np.max(predictions[i])))
plt.imshow(train_images[i])
plt.subplot(2, 2, 4)
plt.bar(range(10), predictions[i])
#plt.bar(range(10), predictions[i])[np.argmax(predictions[i])].set_color('blue')
#plt.bar(range(10), predictions[i])[train_labels[i]].set_color('red')

plt.show()
