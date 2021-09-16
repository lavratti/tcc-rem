import os
import time

import matplotlib
matplotlib.use('Agg') # No pictures displayed
import pylab
import librosa
import librosa.display
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


def gerar_espectograma(amostra):
    print("Gerando espec. para {}".format(amostra))
    matplotlib.use('Agg') # No pictures displayed
    sig, fs = librosa.load(amostra)
    save_path = os.path.join('temp', '{}.jpg'.format(os.path.split(amostra)[1].split('.')[0]))
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    S = librosa.feature.melspectrogram(y=sig, sr=fs)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()
    print("Espec. OK")


class ThreadedProcessarLista(QThread):

    progresso_total = pyqtSignal(int)
    progresso_parcial = pyqtSignal(int)

    def __init__(self, parent=None, amostras=None):
        super(ThreadedProcessarLista, self).__init__(parent)
        if not amostras:
            amostras = []
        self.amostras = amostras
        self.is_running = True

    def run(self):

        amostras_processadas = 0
        self.progresso_total.emit(0)

        for amostra in self.amostras:

            progresso_na_amostra = 0
            self.progresso_parcial.emit(progresso_na_amostra)
            gerar_espectograma(amostra)
            progresso_na_amostra = 50
            self.progresso_parcial.emit(progresso_na_amostra)

            for progresso_na_amostra in range(51,100):
                self.progresso_parcial.emit(progresso_na_amostra+1)
                time.sleep(0.002)

            amostras_processadas += 1
            progresso_na_lista = int(100 * (amostras_processadas / len(self.amostras)))
            self.progresso_total.emit(progresso_na_lista)

        self.progresso_total.emit(100)
        self.is_running = False
