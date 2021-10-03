"""
Thanks @Zain
"""
import csv

import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, Permute, BatchNormalization
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.layers.recurrent import GRU, LSTM
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score


def CRNN2D():
    '''
    Model used for evaluation in paper. Inspired by K. Choi model in:
    https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/music_tagger_crnn.py
    '''

    nb_layers = 4  # number of convolutional layers
    nb_filters = [64, 128, 128, 128]  # filter sizes
    kernel_size = (3, 3)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),
                 (4, 2)]  # size of pooling area

    # shape of input data (frequency, time, channels)
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model and normalize along frequency axis
    model = Sequential()
    model.add(BatchNormalization(axis=frequency_axis))

    # First convolution layer specifies shape
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     ))
    model.add(Activation(activation))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0]))
    model.add(Dropout(0.1))

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(
            axis=channel_axis))  # Improves overfitting/underfitting
        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1]))  # Max pooling
        model.add(Dropout(0.1))

        # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(1))
    return model


def melgram(audio_path):
    ''' Compute a mel-spectrogram and returns it in a shape of (1,1,96,?), where
    96 == #mel-bins and
    parameters
    ----------
    audio_path: path for the audio file.
                Any format supported by audioread will work.
    More info: http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load
    '''

    # mel-spectrogram parameters
    SR = 22050
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    ret = librosa.feature.melspectrogram(y=src, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS) ** 2
    ret = ret[np.newaxis, np.newaxis, :]
    return ret


def main():
    seed = 1

    X = []
    y_arrousal = []
    y_valence = []

    afile = open(
        'C:\\Users\\lucas.lavratti\\OneDrive - Grupo Marista\\Eng. de Computação\\TCC\\tcc-rem\\src\\crnn\\dataset\\annotations\\static_annotations.csv')
    reader = csv.reader(afile)
    next(reader)  # pular header
    for n in range(5):
        song_id, mean_arousal, std_arousal, mean_valence, std_valence = next(reader)
        song_id = int(song_id)
        # shape of input data (frequency, time, channels)
        mel = melgram(
            'C:\\Users\\lucas.lavratti\\OneDrive - Grupo Marista\\Eng. de Computação\\TCC\\tcc-rem\\src\\crnn\\dataset\\clips_45seconds\\{}.mp3'.format(
                song_id))
        X.append(mel)
        y_arrousal.append(float(mean_arousal))
        y_valence.append(float(mean_valence))
        print("{:3.0f}% done reading samples. Song id: {:02d}, valence {:1.3f}, arrousal {:1.3f}.".format(100*(n/50), song_id, y_valence[-1], y_arrousal[-1]))

    estimator = KerasRegressor(build_fn=CRNN2D, verbose=True)
    # kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator, X, y_arrousal)  # , cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    estimator.fit(X, y_arrousal)
    prediction = estimator.predict(X)
    accuracy_score(y_arrousal, prediction)


if __name__ == "__main__":
    main()
