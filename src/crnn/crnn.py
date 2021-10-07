import os

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

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


def build_model():
    model = keras.Sequential([
        # Input block
        layers.BatchNormalization(name='bn0'),
        layers.Reshape((128, 1940, 1)),

        # Conv block 1
        layers.Convolution2D(64, 3, 3, name='conv1'),
        layers.BatchNormalization(axis=3, name='bn1'),
        layers.ELU(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1'),
        layers.Dropout(0.1, name='dropout1'),

        # Conv block 2
        layers.Convolution2D(128, 3, 3, name='conv2'),
        layers.BatchNormalization(axis=3, name='bn2'),
        layers.ELU(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2'),
        layers.Dropout(0.1, name='dropout2'),

        # GRU
        layers.Reshape((3 * 53, 128)),
        layers.GRU(32, return_sequences=True, name='gru1'),
        layers.GRU(32, return_sequences=False, name='gru2'),

        # Out
        layers.Dropout(0.3, name='dropout3'),
        layers.Dense(1, activation='sigmoid', name='output')
    ])

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

model = build_model()
model.build(np.shape(train_dataset))
model.summary()

EPOCHS = 1000
history = model.fit(
    train_dataset, mean_arousals,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
plot_history(history)
