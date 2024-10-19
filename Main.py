import tensorflow as tf
import numpy as np
import pickle
import gzip
import os

import certifi
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist

# Loading the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = Sequential()

# Input Layer
model.add(Flatten(input_shape=(28, 28))) 

# Hidden Layer
model.add(Dense(523, activation='relu'))
model.add(Dense(349, activation='relu'))
model.add(Dense(233, activation='relu'))
model.add(Dense(155, activation='relu'))
model.add(Dense(103, activation='relu'))
model.add(Dense(69, activation='relu'))
model.add(Dense(47, activation='relu'))
model.add(Dense(31, activation='relu'))
model.add(Dense(21, activation='relu'))

# Output Layer
model.add(Dense(10, activation='softmax'))

# Compile and Fit
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 60, batch_size = 64, validation_split = 0.1)

# Predictions
predictions = model.predict(x_test)

count = 0
for i in range(10000):
    if tf.argmax(y_test[i]).numpy() != tf.argmax(predictions[i]).numpy():
        count+=1

print(100 - ((count/10000) * 100))

#Save Model
model.save('trainedModel.h5')