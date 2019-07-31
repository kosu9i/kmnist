#!/usr/bin/env python3
# coding: utf-8

# Based on MNIST CNN from Keras' examples: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py (MIT License)

from __future__ import print_function

import random
import pickle
from pathlib import Path
import sys
from PIL import Image
from tqdm import tqdm

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np

from generator import MyGenerator

from functools import partial

_print = partial(print, file=sys.stderr, flush=True)

_print('load data paths.')

random.seed(0)

jpg_glob_list = list(Path('../../../data/crop/train_crop/').glob('*/*.jpg'))
random.shuffle(jpg_glob_list)

_print('split data paths and labels.')
data_paths, labels = [], []
for i, img_path in enumerate(jpg_glob_list):
#    if i > 10000:
#        break
    data_paths.append(img_path)
    labels.append(img_path.parent.name)

batch_size = 128
num_of_class = len(set(labels))
epochs = 100
channel_num = 1

# input image dimensions
img_height, img_width = 64, 64

if K.image_data_format() == 'channels_first':
    input_shape = (channel_num, img_height, img_width)
else:
    input_shape = (img_height, img_width, channel_num)

_print('label binarize.')
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
labels_one_hot = lb.fit_transform(labels)
with open('./label_binarizer.model', 'wb') as fp:
    pickle.dump(lb, fp)

from sklearn.model_selection import train_test_split

valid_size=0.1
valid_idx = int(len(data_paths) * (1 - valid_size))

#_print('split train and test.')
#X_train, X_valid, y_train, y_valid = train_test_split(
#    data_paths, labels_one_hot, test_size=0.1)

_print('train gen')
train_gen = MyGenerator(data_paths[:valid_idx], labels_one_hot[:valid_idx],
    batch_size=batch_size,
    width=img_width, height=img_height, channel_num=channel_num,
    num_of_class=num_of_class)

_print('valid gen')
valid_gen = MyGenerator(data_paths[valid_idx:], labels_one_hot[valid_idx:], batch_size=batch_size,
    width=img_width, height=img_height, channel_num=channel_num,
    num_of_class=num_of_class)

_print('set session')
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="2", # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

_print('model define')
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_of_class, activation='softmax'))

_print('model compile')
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

checkpoint_path = './model/kaggle_{epoch:02d}-{val_loss:.5f}.hdf5'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=False)

_print('fit generator')
model.fit_generator(train_gen, steps_per_epoch=train_gen.steps_per_epoch,
          validation_data=valid_gen,
          validation_steps=valid_gen.steps_per_epoch,
          epochs=epochs,
          verbose=1,
          callbacks=[checkpoint])

model.save('./model/kaggle_model.hdf5')
train_score = model.evaluate(x_train, y_train, verbose=0)
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', train_score[1])
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])
print(data_paths[:4])

sys.exit(0)


