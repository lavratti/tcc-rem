import os
import time

import matplotlib
matplotlib.use('Agg') # No pictures displayed
import pylab
import librosa.display
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


def gerar_espectograma(amostra):
    print("Gerando espec. para {}".format(amostra))
    matplotlib.use('Agg') # No pictures displayed
    sig, fs = librosa.load(amostra)
    save_path = os.path.join('spectograms', '{}.jpg'.format(os.path.split(amostra)[1].split('.')[0]))
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    S = librosa.feature.melspectrogram(y=sig, sr=fs)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()
    print("Espec. OK")
    return save_path


class ThreadedProcessarLista(QThread):

    progresso_total = pyqtSignal(int)
    progresso_parcial = pyqtSignal(int)

    def __init__(self, parent=None, amostras=None, historico=None):
        super(ThreadedProcessarLista, self).__init__(parent)
        if not amostras or not historico:
            raise ValueError
        self.amostras = amostras
        self.historico = historico
        self.is_running = True

    def run(self):

        amostras_processadas = 0
        self.progresso_total.emit(0)

        for amostra in self.amostras:

            # Atualiza barra de progresso zerando amostra atual
            self.progresso_parcial.emit(0)

            # Processamento inicial
            resultado = processamento_basico_amostra(amostra)
            resultado['spectogram'] = gerar_espectograma(amostra)
            # TODO: Processamento maior + Atualiza barra de progresso

            # Atualiza barra de progresso
            self.progresso_parcial.emit(99)
            self.historico.salvar_no_historico(resultado)

            # Atualiza barra de progresso finalizando a amostra atual
            amostras_processadas += 1
            self.progresso_parcial.emit(100)
            self.progresso_total.emit(int(100 * (amostras_processadas / len(self.amostras))))

        # Finalizou de processar a lista
        self.progresso_total.emit(100)
        self.is_running = False


def processamento_basico_amostra(amostra):
    propriedades = {'arquivo': amostra}
    y, sr = librosa.load(amostra)
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    propriedades['bpm'] = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    return propriedades

