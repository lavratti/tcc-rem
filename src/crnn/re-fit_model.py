import os
import shutil
import sys
import random
import matplotlib.pyplot as plt
import pylab
import dataset.mydataset as mydataset
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

crrn_folder_path = (os.path.dirname(os.path.realpath(__file__)))


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    pylab.xlabel('Epoch')
    pylab.ylabel('Mean Abs Error [MPG]')
    pylab.plot(hist['epoch'], hist['valence_mae'], label='valence_mae')
    pylab.plot(hist['epoch'], hist['val_valence_mae'], label='val_valence_mae')
    pylab.plot(hist['epoch'], hist['arousal_mae'], label='arousal_mae')
    pylab.plot(hist['epoch'], hist['val_arousal_mae'], label='val_arousal_mae')
    pylab.legend()
    pylab.savefig(os.path.join(crrn_folder_path, 'last_result_MAE.png'))
    pylab.close()

    pylab.xlabel('Epoch')
    pylab.ylabel('Mean Square Error [$MPG^2$]')
    pylab.plot(hist['epoch'], hist['valence_mse'], label='valence_mse')
    pylab.plot(hist['epoch'], hist['val_valence_mse'], label='val_valence_mse')
    pylab.plot(hist['epoch'], hist['arousal_mse'], label='arousal_mse')
    pylab.plot(hist['epoch'], hist['val_arousal_mse'], label='val_arousal_mse')
    pylab.legend()
    pylab.savefig(os.path.join(crrn_folder_path, 'last_result_MSE.png'))
    pylab.close()


def plot_prediction_results(test_labels, prediction):
    test_points = []
    pred_points = []
    for i in range(len(prediction[0])):
        test_points.append([test_labels[0][i], test_labels[1][i]])
        pred_points.append([float(prediction[0][i]), float(prediction[1][i])])
    plt.figure()
    for a, b in zip(test_points, pred_points):
        plt.scatter(a[0], a[1], s=1, color="blue")
        plt.scatter(b[0], b[1], s=1, color="orange")
        plt.plot([a[0], b[0]], [a[1], b[1]], "black", linewidth=1, alpha=0.1)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.savefig(os.path.join(crrn_folder_path, "model-last-fit-predict.png"))
    plt.close()


def get_model(shape=(128, 64, 1)):

    inputs = tf.keras.Input(shape)
    
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(inputs)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Reshape((62, -1))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(8))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(8, activation='tanh',))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = [tf.keras.layers.Dense(1, name=name)(x) for name in ['valence', 'arousal']]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])

    return model


def main():

    if not os.path.isfile(os.path.join(crrn_folder_path, "dataset", "melspectograms.pickle")):
        print("Generating pickle")
        mydataset.gen_pickle()
        print("Generating pickle")

    print("Loading pickle")
    mean_arousals, mean_valences, dataset = mydataset.quick_load()
    print("Loaded pickle")
    
    # Manual normalization of dataset
    for aux in range(len(dataset)):
        mean_valences[aux] = (mean_valences[aux])/5 - 1
        mean_arousals[aux] = (mean_arousals[aux])/5 - 1
    
    
    split = 0.6
    train_labels = [mean_valences[:int(len(mean_valences) * split)], mean_arousals[:int(len(mean_arousals) * split)]]
    test_labels = [mean_valences[int(len(mean_valences) * split):], mean_arousals[int(len(mean_arousals) * split):]]
    train_dataset = dataset[:int(len(dataset) * split)]
    test_dataset = dataset[int(len(dataset) * split):]

    print("Samples\nTrain: {}\nTest: {}".format(len(train_dataset), len(test_dataset)))

    if not os.path.isfile(os.path.join(crrn_folder_path, "model", "saved_model.pb")):

        if not os.path.exists(os.path.join(crrn_folder_path, "model")):
            os.mkdir(os.path.join(crrn_folder_path, "model"))

        model = get_model()
        model.build(np.shape(train_dataset))
        model.summary()

        EPOCHS = 500
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='valence_mse', patience=10)
        history = model.fit(
            train_dataset, train_labels,
            epochs=EPOCHS, validation_split=0.2, verbose=0, shuffle=True,
            callbacks=[tfa.callbacks.TQDMProgressBar(show_epoch_progress=False), es_callback])

        model.save(os.path.join(crrn_folder_path, "model"))
        plot_history(history)
    else:
        print("Modelo salvo encontrado.")
        print("Para mudar o modelo (ou re fazer o fit), apague a pasta {}".format(os.path.join(crrn_folder_path, "model")))

    model = tf.keras.models.load_model(os.path.join(crrn_folder_path, "model"))

    weights = model.get_weights()
    single_in_model = get_model()
    single_in_model.set_weights(weights)
    single_in_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(), metrics=['mae', 'mse'])

    prediction = model.predict(test_dataset)

    plot_prediction_results(test_labels, prediction)
    

if __name__ == "__main__":
    main()
