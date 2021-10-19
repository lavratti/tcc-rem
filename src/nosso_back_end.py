import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import librosa.display
import matplotlib
import os
import logging
import tensorflow as tf

matplotlib.use('Agg') # No pictures displayed

#serviço de Log
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(fmt)
logger.addHandler(ch)
fh = logging.FileHandler(os.path.join("..", "rem.log"))
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
logger.addHandler(fh)

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
        logger.info("Thread ThreadedProcessarLista começou a trabalhar")

        amostras_processadas = 0
        self.progresso_total.emit(0)
        for amostra in self.amostras:

            # Atualiza barra de progresso zerando amostra atual
            self.progresso_parcial.emit(0)

            # Processamento inicial
            resultado = processamento_basico_amostra(amostra)
            self.progresso_parcial.emit(10)
            predict = processamento_CRNN_amostra(amostra)
            #resultado['valence'] = 0
            #resultado['arousal'] = 0
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
        logger.info("Thread ThreadedProcessarLista terminou de trabalhar")

        self.is_running = False


def processamento_basico_amostra(amostra):
    logger.info("Chamada processamento_basico_amostra para {}".format(amostra))
    propriedades = {'arquivo': amostra}
    y, sr = librosa.load(amostra)
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    propriedades['bpm'] = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    logger.info("Resultado: {}".format(propriedades))
    return propriedades

def processamento_CRNN_amostra(amostra):
    logger.info("Chamada processamento_CRNN_amostra para {}".format(amostra))
    sig, fs = librosa.load(amostra)
    hl = int(len(sig) / 64)
    melgram = librosa.feature.melspectrogram(y=sig, sr=fs, hop_length=hl,)
    melgram = melgram[:, :64]
    melgram_p = librosa.power_to_db(melgram, ref=np.max)
    melgram_p.reshape(128, 64, 1)
    model = tf.keras.models.load_model("./../crnn/model", compile=True)
    prediction = model.predict(np.array([melgram_p,]), batch_size=1)
    logger.info("Resultado: {}".format(prediction))
    return prediction
