# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 12:49:28 2020

@author: Manik Bali
"""
# Code prints out the number of images correctly classified if all the test images 
#are given  as input.

from mnist import MNIST
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

tf.compat.v1.disable_eager_execution()
#plt.clf()
#Read in MNIST data

mndata = MNIST(r"C:/Users/Manik Bali/Documents/KD/SVM")
mndata.gz=True
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

test_images=np.array(test_images).astype('float32')/255.0 
test_labels=np.array(test_labels) 
# Set Number of training images
Number_of_train_images=10000 
#Read and Load in MNIST data into train_images and train_labels
train_images=np.array(train_images[0:Number_of_train_images]).astype('float32')/255
train_labels=np.array(train_labels[0:Number_of_train_images]) 

#Resize the train images to arrays that can be input into the model function
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images =  test_images.reshape(test_images.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)

# Importing the required Keras modules containing model and layers

# Create a Sequential Model and add the layers
model = tf.keras.Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=train_images,y=train_labels, epochs=10)

model.evaluate(test_images, test_labels)

image_index = 4444
plt.imshow(test_images[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(test_images[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())
