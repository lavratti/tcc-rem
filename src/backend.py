import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import librosa.display
import matplotlib
import os
import logging
import tensorflow as tf
from PIL import Image

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

src_folder_path = (os.path.dirname(os.path.realpath(__file__)))
img_folder_path = os.path.join(src_folder_path, "imgs")

model = tf.keras.models.load_model(os.path.join(src_folder_path, "crnn", "model"))

def make_va_plot(predict):

    tmp_path = os.path.join(img_folder_path, "temp.png")

    plt.figure(figsize=(4.5, 4.5))
    plt.axis('off')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    # linha plt.plot([-1, 1], [0, 0], color="gray")
    # linha plt.plot([0, 0], [-1, 1], color="gray")
    plt.scatter(predict[0], predict[1], s=400, color="red")
    plt.savefig(tmp_path, transparent=True)
    plt.close()
    
    background = Image.open(os.path.join(img_folder_path, 'img1.png'), 'r').convert("RGBA")
    foreground = Image.open(tmp_path, 'r').convert("RGBA")
    bg_w, bg_h = background.size
    img_w, img_h = foreground.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(foreground, offset, mask=foreground)
    background.save(tmp_path)
    
    return tmp_path


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
        self.progresso_total.emit(1)
        for amostra in self.amostras:

            # Atualiza barra de progresso zerando amostra atual
            self.progresso_parcial.emit(1)

            # Processamento inicial
            resultado = processamento_basico_amostra(amostra)
            self.progresso_parcial.emit(20)
            predict = processamento_crnn_amostra(amostra)
            self.progresso_parcial.emit(90)
            resultado['valence'] = float(predict[0][0])
            resultado['arousal'] = float(predict[1][0])
            make_va_plot(predict)
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

def processamento_crnn_amostra(amostra):
    logger.info("Chamada processamento_crnn_amostra para {}".format(amostra))
    sig, fs = librosa.load(amostra)
    hl = int(len(sig) / 64)
    melgram = librosa.feature.melspectrogram(y=sig, sr=fs, hop_length=hl,)
    melgram = melgram[:, :64]
    melgram_p = librosa.power_to_db(melgram, ref=np.max)
    melgram_p = melgram_p.reshape(1, 128, 64, 1)
    prediction = model.predict(melgram_p, batch_size=1)
    logger.info("Resultado: {}".format(prediction))
    return prediction
