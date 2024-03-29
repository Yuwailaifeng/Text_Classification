#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
from keras.callbacks import EarlyStopping
from keras.datasets import imdb
from keras.preprocessing import sequence

from rcnn import RCNN

import keras
from keras import backend
import tensorflow as tf
print(tf.__version__)
print(tf.test.is_gpu_available())
# tf.debugging.set_log_device_placement(True)

gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
config.gpu_options.per_process_gpu_memory_fraction = 0.7
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
session = tf.Session(config=config)



max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 10

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# for i in range(3):
# 	print(x_train[i])
# 	print(y_train[i])

print('Pad sequences (samples x time)...')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Prepare input for model...')
x_train_current = x_train
x_train_left = np.hstack([np.expand_dims(x_train[:, 0], axis=1), x_train[:, 0:-1]])
x_train_right = np.hstack([x_train[:, 1:], np.expand_dims(x_train[:, -1], axis=1)])
x_test_current = x_test
x_test_left = np.hstack([np.expand_dims(x_test[:, 0], axis=1), x_test[:, 0:-1]])
x_test_right = np.hstack([x_test[:, 1:], np.expand_dims(x_test[:, -1], axis=1)])
print('x_train_current shape:', x_train_current.shape)
print('x_train_left shape:', x_train_left.shape)
print('x_train_right shape:', x_train_right.shape)
print('x_test_current shape:', x_test_current.shape)
print('x_test_left shape:', x_test_left.shape)
print('x_test_right shape:', x_test_right.shape)

for i in range(3):
	print(x_train_current[i][-3:])
	print(x_train_left[i][-3:])
	print(x_train_right[i][-3:])
	print()




print('Build model...')
model = RCNN(maxlen, max_features, embedding_dims).get_model()
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
model.fit([x_train_current, x_train_left, x_train_right], y_train,
		  batch_size=batch_size,
		  epochs=epochs,
		  callbacks=[early_stopping],
		  validation_data=([x_test_current, x_test_left, x_test_right], y_test))

print('Test...')
result = model.predict([x_test_current, x_test_left, x_test_right])
