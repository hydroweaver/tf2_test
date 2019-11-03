
#DONT FORGET TO UNDERSTAND THE CODE, LIKE SWITCHING TEST FOR TRAIN WHILE LOSING YOURSELF INTO MATPLOTLIB!!!!! THATS JUST PLAIN STUPID !

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

<<<<<<< HEAD
#main_dir =

#'''train_images = np.load(r'C:\Users\hydro\.spyder-py3\tf2\train_images.npy')
#train_labels = np.load(r'C:\Users\hydro\.spyder-py3\tf2\train_labels.npy')
#predictions = np.load(r'C:\Users\hydro\.spyder-py3\tf2\predictions.npy')'''



train_images = np.load(r'C:\Users\Karan.Verma\.spyder-py3\tf2\train_images.npy')
train_labels = np.load(r'C:\Users\Karan.Verma\.spyder-py3\tf2\train_labels.npy')
predictions = np.load(r'C:\Users\Karan.Verma\.spyder-py3\tf2\predictions.npy')
=======
test_images = np.load(r'C:\Users\hydro\.spyder-py3\tf2\test_images.npy')
test_labels = np.load(r'C:\Users\hydro\.spyder-py3\tf2\test_labels.npy')
predictions = np.load(r'C:\Users\hydro\.spyder-py3\tf2\predictions.npy')
>>>>>>> be11583aa0370843d361ab6c63c4457458f74034

#test_labels = tf.one_hot(test_labels, 10)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_image(i, class_names, test_labels, predictions):
    test_label_text, test_label, prediction = class_names[test_labels[i]], test_labels[i], predictions
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    predicted_label = np.argmax(prediction)

    if predicted_label == test_label:
        color = 'blue'
    else:
        color = 'red'

    
    plt.xlabel('Original {} Predicted {} {:2.0f}% Acc'.\
               format(class_names[test_labels[i]], class_names[np.argmax(prediction)],\
                      100*np.max(prediction), colors = color))
    plt.imshow(test_images[i], cmap=plt.cm.binary)

def plot_barch(i, predictions, test_labels):
    prediction, test_label = predictions, test_labels[i]
    plt.xticks(range(10))
    plt.yticks([])
    plt.grid(False)
    plt.ylim([0, 1])
    thisplot = plt.bar(range(10), prediction)

    predicted_label = np.argmax(prediction)

    thisplot[predicted_label].set_color('red')
    thisplot[test_label].set_color('blue')

                                                            
num_rows = 5
num_cols = 3

num_images = num_rows * num_cols
#plt.subplots(1, 2)
plt.figure(figsize=(2*2*num_cols, 2*num_rows ))

for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, class_names, test_labels, predictions[i])
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_barch(i, predictions[i], test_labels)

plt.tight_layout()
plt.show()


model = tf.keras.models.load_model(r'C:\Users\hydro\.spyder-py3\tf2\fashion_mnist\model.h5')

i = 755
example_image_to_be_predicted = test_images[i]
example_image_to_be_predicted_expanded = (np.expand_dims(example_image_to_be_predicted, 0))

single_prediction = model.predict(example_image_to_be_predicted_expanded)

plot_barch(i, single_prediction[0], test_labels)
plt.xticks(range(10), class_names, rotation = 45)
plt.show()





