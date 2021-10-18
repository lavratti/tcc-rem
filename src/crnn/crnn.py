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

random.seed(1)

activation = 'elu' #'relu' #'selu'

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
    pylab.savefig('last_result_MAE.png')
    pylab.close()

    pylab.xlabel('Epoch')
    pylab.ylabel('Mean Square Error [$MPG^2$]')
    pylab.plot(hist['epoch'], hist['valence_mse'], label='valence_mse')
    pylab.plot(hist['epoch'], hist['val_valence_mse'], label='val_valence_mse')
    pylab.plot(hist['epoch'], hist['arousal_mse'], label='arousal_mse')
    pylab.plot(hist['epoch'], hist['val_arousal_mse'], label='val_arousal_mse')
    pylab.legend()
    pylab.savefig('last_result_MSE.png')
    pylab.close()


def get_model():

    inputs = tf.keras.Input(shape=(128, 64, 1))

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(inputs)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x) # 1 layer
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    # x = tf.keras.layers.AveragePooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    x = tf.keras.layers.Dropout(0.75)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation=activation, padding='same')(x) # 2 layer
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    # x = tf.keras.layers.AveragePooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    x = tf.keras.layers.Dropout(0.75)(x)

    # x = tf.keras.layers.Conv2D(64, (3, 3), activation=activation, padding='same')(x) # 3 layer
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    # x = tf.keras.layers.AveragePooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    #x = tf.keras.layers.Dropout(0.75)(x)

    x = tf.keras.layers.Reshape((62, -1))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(8))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(8, activation='tanh',padding='same'))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    outputs = [tf.keras.layers.Dense(1, name=name)(x) for name in ['valence', 'arousal']]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])

    return model


def main():
    if len(sys.argv) >= 2:
        print(sys.argv)
        if sys.argv[1] == "-n":
            print("Are you sure you whant to refit the model? (y/n)")
            confirm = input()
            if confirm == "y":
                shutil.rmtree("model")
                os.mkdir("model")
            else:
                print("Aborting")
                exit()

    if not os.path.isfile('dataset/melspectograms.pickle'):
        print("Generating pickle")
        mydataset.gen_pickle()
        print("Generating pickle")

    print("Loading pickle")
    mean_arousals, mean_valences, dataset = mydataset.quick_load()
    print("Loaded pickle")

    for aux in range(len(dataset)):
        mean_valences[aux] = (mean_valences[aux])/5 - 1
        mean_arousals[aux] = (mean_arousals[aux])/5 - 1

    split = 0.5
    train_labels = [mean_valences[:int(len(mean_valences) * split)], mean_arousals[:int(len(mean_arousals) * split)]]
    test_labels = [mean_valences[int(len(mean_valences) * split):], mean_arousals[int(len(mean_arousals) * split):]]
    train_dataset = dataset[:int(len(dataset) * split)]
    test_dataset = dataset[int(len(dataset) * split):]

    print("Train: {}; Test: {}".format(len(train_dataset), len(test_dataset)))

    if not os.path.isfile("model/saved_model.pb"):
        model = get_model()
        model.build(np.shape(train_dataset))
        model.summary()

        EPOCHS = 1000
        # es_callback = tf.keras.callbacks.EarlyStopping(monitor='mse', patience=10)
        history = model.fit(
            train_dataset, train_labels,
            epochs=EPOCHS, validation_split=0.2, verbose=0, shuffle=True,
            callbacks=[tfa.callbacks.TQDMProgressBar(show_epoch_progress=False), ])  # es_callback])

        model.save("model")
        plot_history(history)

    model = tf.keras.models.load_model("model")

    print("Evaluate")
    result = model.evaluate(test_dataset, test_labels)
    print(dict(zip(model.metrics_names, result)))

    prediction = model.predict(test_dataset)
    test_points = []
    pred_points = []
    # print("True labels | Predicted labels")
    for i in range(len(prediction[0])):
        test_points.append([test_labels[0][i], test_labels[1][i]])
        pred_points.append([float(prediction[0][i]), float(prediction[1][i])])

    plt.scatter(*zip(*test_points), s=1)
    plt.scatter(*zip(*pred_points), s=1)
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.savefig("last_result.png")
    plt.close()

    plt.figure()
    for a, b in zip(test_points, pred_points):
        plt.scatter(a[0], a[1], s=1, color="blue")
        plt.scatter(b[0], b[1], s=1, color="orange")
        plt.plot([a[0], b[0]], [a[1], b[1]], "black", linewidth=1, alpha=0.1)
    plt.show()

if __name__ == "__main__":
    main()
