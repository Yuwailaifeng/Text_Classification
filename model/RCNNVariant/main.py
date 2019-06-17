#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from keras.callbacks import EarlyStopping
from keras.datasets import imdb
from keras.preprocessing import sequence

from rcnn_variant import RCNNVariant

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

print('Pad sequences (samples x time)...')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = RCNNVariant(maxlen, max_features, embedding_dims).get_model()
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
model.fit(x_train, y_train,
		  batch_size=batch_size,
		  epochs=epochs,
		  callbacks=[early_stopping],
		  validation_data=(x_test, y_test))

print('Test...')
result = model.predict(x_test)
