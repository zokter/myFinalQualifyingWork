# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 01:23:45 2020

@author: Андрей Симуков
"""

import numpy
# from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

classCount = 6
epochs = 6
batch_size = 20
nb_trin_samples = 9252
nb_validation_samples = 900
nb_test_samples = 900
img_width, img_height = 224, 224

#fix

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus


#seed for repeatability results

numpy.random.seed(42)




# #download dataset CIFAR-10

# (X_train, y_train), (X_test, y_test) = cifar10.load_data()

# #normalize dataset of pixel intensity

# X_trin = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_trin /= 255
# X_test /= 255

# #transform class tags to categories

# Y_train = np_utils.to_categorical(y_train, 10)
# Y_test = np_utils.to_categorical(y_test, 10)




#create model 

model = Sequential()

#first convolutional layer 
#this applies 32 convolution filters of size 3x3 each.

model.add(Convolution2D(32, (3, 3), border_mode='same', input_shape=(224, 224, 3), activation='relu', dim_ordering="th"))
model.add(Convolution2D(32, (3, 3), border_mode='same', activation='relu'))
model.add(Convolution2D(32, (3, 3), border_mode='same', activation='relu'))

#subsampling layer
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

#regulation layer

model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), border_mode='same', activation='relu'))
model.add(Convolution2D(64, (3, 3), border_mode='same', activation='relu'))
model.add(Convolution2D(64, (3, 3), border_mode='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))


model.add(Dropout(0.25))

# could be more layers

model.add(Convolution2D(128, (3, 3), border_mode='same', activation='relu'))
model.add(Convolution2D(128, (3, 3), border_mode='same', activation='relu'))
model.add(Convolution2D(128, (3, 3), border_mode='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

model.add(Dropout(0.25))

model.add(Convolution2D(256, (3, 3), border_mode='same', activation='relu'))
model.add(Convolution2D(256, (3, 3), border_mode='same', activation='relu'))
model.add(Convolution2D(256, (3, 3), border_mode='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

model.add(Dropout(0.25))

#transform from 2D to 3D tensor

model.add(Flatten())

#dense layer 

model.add(Dense(512, activation='relu'))

#regulation layer

model.add(Dropout(0.25))

#output layer

model.add(Dense(classCount, activation='softmax'))

#compile model

model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
# model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, validation_split=0.1, shuffle=True) #for cifar10 

train_dir = 'KerasModel/train'
val_dir = 'KerasModel/val'
test_dir = 'KerasModel/test'
#bacrend tensorflow, chanels last
input_shape = (img_width, img_height, 3)

# data generator on class ImageDataGenerator. 
datagen = ImageDataGenerator(rescale=1./ 255) 

train_generator = datagen.flow_from_directory(
    train_dir, #folder thet obtein 
    target_size=(img_width, img_height),
    batch_size = batch_size,
    class_mode = "categorical"
)

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size = batch_size,
    class_mode = "categorical"
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size = batch_size,
    class_mode = "categorical"
)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_trin_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size
)

model.save("MyOwnModel.h5")
scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Accuracy on test data: %.2f%%" % (scores[1]*100))
input()