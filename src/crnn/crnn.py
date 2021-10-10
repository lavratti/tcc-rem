import os

import pylab
from tensorflow import keras
from tensorflow.keras import layers
import dataset.mydataset as mydataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    pylab.xlabel('Epoch')
    pylab.ylabel('Mean Abs Error [MPG]')
    pylab.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    pylab.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    pylab.ylim([0, 5])
    pylab.legend()
    pylab.savefig('last_result_MPG.png')
    pylab.close()

    pylab.xlabel('Epoch')
    pylab.ylabel('Mean Square Error [$MPG^2$]')
    pylab.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    pylab.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    pylab.ylim([0, 20])
    pylab.legend()
    pylab.savefig('last_result_MPG2.png')
    pylab.close()


def get_model():

    inputs = tf.keras.Input(shape=(128, 1940, 1))

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 28), strides=(2, 30))(inputs)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(8))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(x)
    x = tf.keras.layers.Reshape(tuple([y for y in x.shape.as_list() if y != 1 and y is not None]))(x)
    x = tf.keras.layers.GRU(8)(x)

    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model

if not os.path.isfile('dataset/melspectograms.pickle'):
    print("Generating pickle")
    mydataset.gen_pickle()
    print("Generating pickle")

print("Loading pickle")
mean_arousals, mean_valences, train_dataset = mydataset.quick_load_500()
print("Loaded pickle")

model = get_model()
model.build(np.shape(train_dataset))
model.summary()

EPOCHS = 10
history = model.fit(
    train_dataset, mean_arousals,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
plot_history(history)
