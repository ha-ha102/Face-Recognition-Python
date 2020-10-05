# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 09:03:47 2018

@author: Home
"""

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()
classifier.add(ZeroPadding2D((1,1),input_shape=(64,64,3)))
classifier.add(Convolution2D(64, (3, 3), activation="relu"))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Convolution2D(64, (3, 3), activation="relu"))
classifier.add(MaxPooling2D((2,2), strides=(2,2)))

classifier.add(ZeroPadding2D((1,1)))
classifier.add(Convolution2D(128, (3, 3), activation='relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Convolution2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D((2,2), strides=(2,2)))

classifier.add(ZeroPadding2D((1,1)))
classifier.add(Convolution2D(256, (3, 3), activation='relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Convolution2D(256, (3, 3), activation='relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Convolution2D(256, (3, 3), activation='relu'))
classifier.add(MaxPooling2D((2,2), strides=(2,2)))

classifier.add(ZeroPadding2D((1,1)))
classifier.add(Convolution2D(512, (3, 3), activation='relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Convolution2D(512, (3, 3), activation='relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Convolution2D(512, (3, 3), activation='relu'))
classifier.add(MaxPooling2D((2,2), strides=(2,2)))

classifier.add(ZeroPadding2D((1,1)))
classifier.add(Convolution2D(512, (3, 3), activation='relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Convolution2D(512, (3, 3), activation='relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Convolution2D(512, (3, 3), activation='relu'))
classifier.add(MaxPooling2D((2,2), strides=(2,2)))

classifier.add(Flatten())
classifier.add(Dense(4096, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(4096, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
classifier.compile(optimizer=sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('Dataset/training_set',#input folder
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'Dataset/test_set', #path to test set
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=1537, #insert number of images in training set
        epochs=1,
        validation_data=test_set,
        validation_steps=len(test_set)//32) #insert number of images in test set
