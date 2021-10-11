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
    pylab.legend()
    pylab.ylim([0, 5])
    pylab.savefig('last_result_MPG2.png')
    pylab.close()


def get_model():
    inputs = tf.keras.Input(shape=(128, 1940, 1))

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 28), strides=(2, 30))(inputs)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(8))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(8))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(x)
    x = tf.keras.layers.Reshape(tuple([y for y in x.shape.as_list() if y != 1 and y is not None]))(x)
    x = tf.keras.layers.GRU(64)(x)

    outputs = tf.keras.layers.Dense(2)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

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

    train_labels = [mean_arousals[:int(len(mean_arousals) * 0.5)], mean_valences[:int(len(mean_valences) * 0.5)]]
    test_labels = [mean_arousals[int(len(mean_arousals) * 0.5):], mean_valences[int(len(mean_valences) * 0.5):]]
    train_dataset = dataset[:int(len(dataset) * 0.5)]
    test_dataset = dataset[int(len(dataset) * 0.5):]

    if not os.path.isfile("model/saved_model.pb"):
        model = get_model()
        model.build(np.shape(train_dataset))
        model.summary()

        EPOCHS = 1000
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        history = model.fit(
            train_dataset, train_labels,
            epochs=EPOCHS, validation_split=0.2, verbose=0, shuffle=True,
            callbacks=[tfa.callbacks.TQDMProgressBar(show_epoch_progress=False), es_callback])

        model.save("model")

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        plot_history(history)  

    model = tf.keras.models.load_model("model")
    prediction = model.predict(test_dataset)
    test_points = []
    pred_points = []
    print("True labels | Predicted labels")
    for i in range(len(test_labels[0])):
        test_points.append([test_labels[0][i], test_labels[1][i]])
        pred_points.append([float(prediction[i][0]), float(prediction[i][1])])
        print("{:2.3f} {:2.3f} | {:2.3f} {:2.3f}".format(test_labels[0][i],
                                                         test_labels[1][i],
                                                         float(prediction[i][0]),
                                                         float(prediction[i][1])))

    plt.scatter(*zip(*test_points), s=1)
    plt.scatter(*zip(*pred_points), s=1)
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.savefig("last_result.png")
    plt.close()

    plt.figure()
    for a, b in zip(test_points, pred_points):
        plt.plot([a[0], b[0]], [a[1], b[1]])
    plt.show()

if __name__ == "__main__":
    main()
