import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed
import pylab
import librosa
import librosa.display
import numpy as np


def gerar_espectograma(amostra):
    print("gerando espc")
    matplotlib.use('Agg') # No pictures displayed
    sig, fs = librosa.load(amostra)
    save_path = os.path.join('temp', '{}.jpg'.format(os.path.split(amostra)[1].split('.')[0]))
    print(save_path)
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    S = librosa.feature.melspectrogram(y=sig, sr=fs)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()