import csv
import sys
from datetime import datetime

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("Done importing python libs.")

print("Loading labels")
song_ids = []
mean_arousals = []
mean_valences = []
with open('dataset/annotations/static_annotations.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)
    for row in reader:
        song_ids.append(int(row[0]))
        mean_arousals.append(float(row[1]))
        mean_valences.append(float(row[3]))

print("Done loading labels")

print("Loading dataset mel-spectograms.")
melspectograms_list = []
t0 = datetime.now()
for n in tqdm(range(20), unit='files', file=sys.stdout):
    if n in song_ids:
        with open('dataset/melgrams/melgram_power_to_db/{}.csv'.format(n), 'rb') as f:
            a = np.loadtxt(f, delimiter=',')
            a = np.resize(a, (128, 1940))
            melspectograms_list.append(a)
    else:
        print("Song not on label list, ignored")

for s in melspectograms_list:
    for r in s:
        print(len(r))
#npa = np.array(melspectograms_list, dtype=np.float)
#print(np.shape(npa))

print("Done loading spectograms into list.")

dataset = pd.DataFrame({'melgram': np.asarray(melspectograms_list).astype(np.float32),
                        'arousal': mean_arousals[:len(melspectograms_list)],
                        'valence': mean_valences[:len(melspectograms_list)]})

print(dataset)

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_arousal_labels = train_dataset.pop('arousal')
train_valence_labels = train_dataset.pop('valence')
test_arousal_labels = test_dataset.pop('arousal')
test_valence_labels = test_dataset.pop('valence')


def basic_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

model = basic_model()
model.summary()

# Mostra o progresso do treinamento imprimindo um único ponto para cada epoch completada
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
EPOCHS = 1000
history = model.fit(
  train_dataset, train_arousal_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
