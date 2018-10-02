# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 14:02:07 2018

@author: Junaid.raza
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#Will be importing 4 numy arrays. images data
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#give Lables to these numpy arrays to plot 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#there are sixty thousands images in trainign set
print (train_images.shape)

print (len(train_labels))

#Train labels are from 0 to 9
print (train_labels)

#Here we have 10 thousands images with 28*28 pixels
print (test_images.shape)

#Test set contains 10k images
print (len(test_labels))

#Preprocess the data
#Showing first image
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

#We scale these values to a range of 0 to 1 before feeding to the neural network model. 
#For this, cast the datatype of the image components from an integer to a float, and divide by 255.
train_images = train_images / 255.0
test_images = test_images / 255.0



#Display the first 25 images from the training set and display the class name below each image.
#Verify that the data is in the correct format and we're ready to build and train the network
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

  

#Building the neural network requires configuring the layers of the model, 
#then compiling the model.
"""
The basic building block of a neural network is the layer. Layers extract representations 
from the data fed into them. And, hopefully, these representations are more meaningful 
for the problem at hand.

Most of deep learning consists of chaining together simple layers. Most layers, 
like tf.keras.layers.Dense, have parameters that are learned during training.

"""

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


"""
The first layer in this network, tf.keras.layers.Flatten, transforms the format 
of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels.

After the pixels are flattened, the network consists of a sequence of two 
tf.keras.layers.Dense layers. These are densely-connected, or fully-connected, 
neural layers. The first Dense layer has 128 nodes (or neurons). 
The second (and last) layer is a 10-node softmax layer—this returns an array of 
10 probability scores that sum to 1. Each node contains a score that indicates the 
probability that the current image belongs to one of the 10 classes.


Compile the model
Before the model is ready for training, it needs a few more settings. These are added 
during the model's compile step:

    Loss function      —This measures how accurate the model is during training. We want to 
        minimize this function to "steer" the model in the right direction.
    Optimizer         —This is how the model is updated based on the data it sees and its loss function.
    Metrics          —Used to monitor the training and testing steps. The following example uses 
        accuracy, the fraction of the images that are correctly classified.
"""
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

#Evaluate accuracy
#Next, compare how the model performs on the test dataset:
test_loss, test_acc = model.evaluate(test_images, test_labels)

print ('Test accuracy:', test_acc)

predictions = model.predict(test_images)
print (predictions[0])
print (np.argmax(predictions[0]))
print (test_labels[0])

#graphs thsi to look full 10 channels
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
#Lets look at the zero index of prediction array
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

#Try another one
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

"""
Let's plot several images with their predictions. Correct prediction labels are blue and 
incorrect prediction labels are red. The number gives the percent (out of 100) for the 
predicted label. Note that it can be wrong even when very confident.
"""
# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
  
#Finally, use the trained model to make a prediction about a single image
# Grab an image from the test dataset
img = test_images[0]
print(img.shape)
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])













"""
Link to this tutorial
https://www.tensorflow.org/tutorials/keras/basic_classification

"""











