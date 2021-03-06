from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMessageBox, QListWidgetItem
import os
from matplotlib import pyplot as plt
import popup_progresso
import os.path, sys
import handler_historico, backend

#serviço de Log
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(fmt)
logger.addHandler(ch)
fh = logging.FileHandler(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "rem_log.txt"))
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(ch)


src_folder_path = os.path.dirname(os.path.realpath(__file__))
img_folder_path = os.path.join(src_folder_path, "imgs")
va_space_bg_img_path = os.path.join(img_folder_path, "img1.png")

def formatar_resultado(dict_resultado):
    # Formatador de dict para string
    s = ""
    for key in dict_resultado:
        s += "{}: {}\n\n".format(key, dict_resultado[key])
    return s


class Ui_Form(object):

    def __init__(self, Form):

        # Declaração de variaveis do programa para acessar via objeto do gui
        self.proxima_tela = 0
        try:
            self.historico = handler_historico.Historico(os.path.join(src_folder_path, "historico.json"))
        except:
            logging.error("Erro ao carregar histórico.")
        with open(os.path.join(src_folder_path, 'sobre.html'), 'r') as fp:
            self.texto_sobre = fp.read()

        # =============================================================================================================
        # Inicialização do GUI!
        # =============================================================================================================

        # =============================================================================================================
        # Constroi a janela/form em sí
        # =============================================================================================================
        Form.setObjectName("Form")
        Form.resize(800, 600)
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(12)
        Form.setFont(font)
        Form.setWindowTitle("CRNN-REM")
        self.stackedWidget = QtWidgets.QStackedWidget(Form)
        self.stackedWidget.setEnabled(True)
        self.stackedWidget.setGeometry(QtCore.QRect(0, -10, 800, 600))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stackedWidget.sizePolicy().hasHeightForWidth())
        self.stackedWidget.setSizePolicy(sizePolicy)
        self.stackedWidget.setObjectName("stackedWidget")

        # =============================================================================================================
        # Constroi a tela inicial
        # =============================================================================================================
        self.tela_inicial = QtWidgets.QWidget()
        self.tela_inicial.setObjectName("tela_inicial")
        self.tela_inicial_label_1 = QtWidgets.QLabel(self.tela_inicial)
        self.tela_inicial_label_1.setGeometry(QtCore.QRect(0, 180, 800, 50))
        self.tela_inicial_label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.tela_inicial_label_1.setObjectName("tela_inicial_label_1")
        self.tela_inicial_label_2 = QtWidgets.QLabel(self.tela_inicial)
        self.tela_inicial_label_2.setGeometry(QtCore.QRect(0, 400, 800, 101))
        self.tela_inicial_label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.tela_inicial_label_2.setWordWrap(True)
        self.tela_inicial_label_2.setObjectName("tela_inicial_label_2")
        self.tela_inicial_splitter = QtWidgets.QSplitter(self.tela_inicial)
        self.tela_inicial_splitter.setGeometry(QtCore.QRect(100, 300, 600, 50))
        self.tela_inicial_splitter.setOrientation(QtCore.Qt.Horizontal)
        self.tela_inicial_splitter.setObjectName("tela_inicial_splitter")
        self.tela_inicial_botao_reconhecer = QtWidgets.QPushButton(self.tela_inicial_splitter)
        self.tela_inicial_botao_reconhecer.setObjectName("tela_inicial_botao_reconhecer")
        self.tela_inicial_botao_historico = QtWidgets.QPushButton(self.tela_inicial_splitter)
        self.tela_inicial_botao_historico.setObjectName("tela_inicial_botao_historico")
    
        self.tela_inicial_botao_sobre = QtWidgets.QPushButton(self.tela_inicial_splitter)
        self.tela_inicial_botao_sobre.setObjectName("tela_inicial_botao_sobre")
        self.stackedWidget.addWidget(self.tela_inicial)
        # Textos da tela
        self.tela_inicial_label_1.setText("Classificação de emoções em músicas utilizando aprendizado de máquina")
        self.tela_inicial_label_2.setText("Lucas Lavratti; Rafael B. G. Bueno; Prof. Dr. Alceu S. Britto.\n"
                                          "Escola Politécnica da Pontifícia Universidade Católica do Paraná (PUC-PR)\n"
                                          "Curitiba, 2021")
        self.tela_inicial_botao_reconhecer.setText("Reconhecer")
        self.tela_inicial_botao_historico.setText("Histórico")
        self.tela_inicial_botao_sobre.setText("Sobre")

        # =============================================================================================================
        # Constroi a tela sobre
        # =============================================================================================================
        self.tela_sobre = QtWidgets.QWidget()
        self.tela_sobre.setObjectName("tela_sobre")
        self.tela_sobre_textBrowser = QtWidgets.QTextBrowser(self.tela_sobre)
        self.tela_sobre_textBrowser.setGeometry(QtCore.QRect(10, 70, 781, 531))
        self.tela_sobre_textBrowser.setObjectName("tela_sobre_textBrowser")
        self.tela_sobre_label_1 = QtWidgets.QLabel(self.tela_sobre)
        self.tela_sobre_label_1.setGeometry(QtCore.QRect(10, 20, 771, 41))
        self.tela_sobre_label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.tela_sobre_label_1.setObjectName("tela_sobre_label_1")
        self.tela_sobre_botao_voltar = QtWidgets.QPushButton(self.tela_sobre)
        self.tela_sobre_botao_voltar.setGeometry(QtCore.QRect(10, 20, 101, 41))
        self.tela_sobre_botao_voltar.setObjectName("tela_sobre_botao_voltar")
        self.stackedWidget.addWidget(self.tela_sobre)
        # Textos da tela
        self.tela_sobre_textBrowser.setHtml(self.texto_sobre)
        self.tela_sobre_label_1.setText("Sobre o projeto")
        self.tela_sobre_botao_voltar.setText("Voltar")

        # =============================================================================================================
        # Constroi a tela reconhecer
        # =============================================================================================================
        self.tela_reco = QtWidgets.QWidget()
        self.tela_reco.setObjectName("tela_reco")
        self.tela_reco_botao_voltar = QtWidgets.QPushButton(self.tela_reco)
        self.tela_reco_botao_voltar.setGeometry(QtCore.QRect(10, 20, 101, 41))
        self.tela_reco_botao_voltar.setObjectName("tela_reco_botao_voltar")
        self.tela_reco_label_1 = QtWidgets.QLabel(self.tela_reco)
        self.tela_reco_label_1.setGeometry(QtCore.QRect(10, 20, 771, 41))
        self.tela_reco_label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.tela_reco_label_1.setObjectName("tela_reco_label_1")
        self.tela_reco_botao_add_amostra = QtWidgets.QPushButton(self.tela_reco)
        self.tela_reco_botao_add_amostra.setGeometry(QtCore.QRect(10, 250, 191, 41))
        self.tela_reco_botao_add_amostra.setObjectName("tela_reco_botao_add_amostra")
        self.tela_reco_listWidget = QtWidgets.QListWidget(self.tela_reco)
        self.tela_reco_listWidget.setGeometry(QtCore.QRect(10, 300, 781, 251))
        self.tela_reco_listWidget.setAlternatingRowColors(True)
        self.tela_reco_listWidget.setObjectName("tela_reco_listWidget")
        self.tela_reco_listWidget.setCurrentRow(-1)
        self.tela_reco_botao_reconhecer = QtWidgets.QPushButton(self.tela_reco)
        self.tela_reco_botao_reconhecer.setGeometry(QtCore.QRect(640, 560, 151, 41))
        self.tela_reco_botao_reconhecer.setObjectName("tela_reco_botao_reconhecer")
        self.tela_reco_label_2 = QtWidgets.QLabel(self.tela_reco)
        self.tela_reco_label_2.setGeometry(QtCore.QRect(20, 70, 761, 171))
        self.tela_reco_label_2.setAlignment(QtCore.Qt.AlignJustify | QtCore.Qt.AlignVCenter)
        self.tela_reco_label_2.setWordWrap(True)
        self.tela_reco_label_2.setObjectName("tela_reco_label_2")
        self.tela_reco_label_1.raise_()
        self.tela_reco_botao_voltar.raise_()
        self.tela_reco_botao_add_amostra.raise_()
        self.tela_reco_listWidget.raise_()
        self.tela_reco_botao_reconhecer.raise_()
        self.tela_reco_label_2.raise_()
        self.stackedWidget.addWidget(self.tela_reco)
        # Textos da tela
        self.tela_reco_botao_voltar.setText("Voltar")
        self.tela_reco_label_1.setText("Reconhecer")
        self.tela_reco_botao_add_amostra.setText("Adicionar amostra")
        self.tela_reco_botao_reconhecer.setText("Reconhecer")
        self.tela_reco_label_2.setText("TEXTO RECONHECER\n"
                                       "Amostras duplicadas não serão selecionadas\n"
                                       "Para remover uma amostra clique duas vez no item da lista.")

        # =============================================================================================================
        # Constroi a tela resultado
        # =============================================================================================================
        self.tela_resultado = QtWidgets.QWidget()
        self.tela_resultado.setObjectName("tela_resultado")
        self.tela_resultado_label_1 = QtWidgets.QLabel(self.tela_resultado)
        self.tela_resultado_label_1.setGeometry(QtCore.QRect(0, 20, 800, 40))
        self.tela_resultado_label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.tela_resultado_label_1.setObjectName("tela_resultado_label_1")
        self.tela_resultado_text_browser = QtWidgets.QTextBrowser(self.tela_resultado)
        self.tela_resultado_text_browser.setGeometry(QtCore.QRect(10, 70, 311, 531))
        self.tela_resultado_text_browser.setObjectName("tela_resultado_text_browser")
        self.tela_resultado_label_figura = QtWidgets.QLabel(self.tela_resultado)
        self.tela_resultado_label_figura.setGeometry(QtCore.QRect(330, 75, 450, 470))
        self.tela_resultado_label_figura.setText("")
        self.tela_resultado_label_figura.setPixmap(QtGui.QPixmap(va_space_bg_img_path))
        self.tela_resultado_label_figura.setScaledContents(False)
        self.tela_resultado_label_figura.setObjectName("tela_resultado_label_figura")
        self.tela_resultado_splitter = QtWidgets.QSplitter(self.tela_resultado)
        self.tela_resultado_splitter.setGeometry(QtCore.QRect(330, 560, 461, 40))
        self.tela_resultado_splitter.setOrientation(QtCore.Qt.Horizontal)
        self.tela_resultado_splitter.setObjectName("tela_resultado_splitter")
        self.tela_resultado_botao_voltar = QtWidgets.QPushButton(self.tela_resultado_splitter)
        self.tela_resultado_botao_voltar.setObjectName("tela_resultado_botao_voltar")
        self.stackedWidget.addWidget(self.tela_resultado)
        # Textos da tela
        self.tela_resultado_label_1.setText("Resultado")
        self.tela_resultado_text_browser.setHtml("")
        self.tela_resultado_botao_voltar.setText("Voltar")

        # =============================================================================================================
        # Constroi a tela historico
        # =============================================================================================================
        self.tela_historico = QtWidgets.QWidget()
        self.tela_historico.setObjectName("tela_historico")
        self.tela_historico_label_1 = QtWidgets.QLabel(self.tela_historico)
        self.tela_historico_label_1.setGeometry(QtCore.QRect(0, 20, 800, 40))
        self.tela_historico_label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.tela_historico_label_1.setObjectName("tela_historico_label_1")
        self.tela_historico_label_figura = QtWidgets.QLabel(self.tela_historico)
        self.tela_historico_label_figura.setGeometry(QtCore.QRect(330, 64, 450, 481))
        self.tela_historico_label_figura.setText("")
        self.tela_resultado_label_figura.setPixmap(QtGui.QPixmap(va_space_bg_img_path))
        self.tela_historico_label_figura.setScaledContents(False)
        self.tela_historico_label_figura.setObjectName("tela_historico_label_figura")
        self.tela_historico_splitter_listas = QtWidgets.QSplitter(self.tela_historico)
        self.tela_historico_splitter_listas.setGeometry(QtCore.QRect(10, 60, 321, 491))
        self.tela_historico_splitter_listas.setOrientation(QtCore.Qt.Vertical)
        self.tela_historico_splitter_listas.setObjectName("tela_historico_splitter_listas")
        self.tela_historico_listWidget = QtWidgets.QListWidget(self.tela_historico_splitter_listas)
        self.tela_historico_listWidget.setAlternatingRowColors(False)
        self.tela_historico_listWidget.setViewMode(QtWidgets.QListView.ListMode)
        self.tela_historico_listWidget.setObjectName("tela_historico_listWidget")
        self.tela_historico_listWidget.setCurrentRow(-1)
        self.tela_historico_textBrowser = QtWidgets.QTextBrowser(self.tela_historico_splitter_listas)
        self.tela_historico_textBrowser.setObjectName("tela_historico_textBrowser")
        self.tela_historico_splitter_botoes = QtWidgets.QSplitter(self.tela_historico)
        self.tela_historico_splitter_botoes.setGeometry(QtCore.QRect(10, 560, 781, 41))
        self.tela_historico_splitter_botoes.setOrientation(QtCore.Qt.Horizontal)
        self.tela_historico_splitter_botoes.setObjectName("tela_historico_splitter_botoes")
        self.tela_historico_botao_limpar_selecao = QtWidgets.QPushButton(self.tela_historico_splitter_botoes)
        self.tela_historico_botao_limpar_selecao.setObjectName("tela_historico_botao_limpar_selecao")
        self.tela_historico_botao_exportar = QtWidgets.QPushButton(self.tela_historico_splitter_botoes)
        self.tela_historico_botao_exportar.setObjectName("tela_historico_botao_exportar")
        self.tela_historico_botao_importar = QtWidgets.QPushButton(self.tela_historico_splitter_botoes)
        self.tela_historico_botao_importar.setObjectName("tela_historico_botao_importar")
        self.tela_historico_botao_voltar = QtWidgets.QPushButton(self.tela_historico_splitter_botoes)
        self.tela_historico_botao_voltar.setObjectName("tela_historico_botao_voltar")
        self.stackedWidget.addWidget(self.tela_historico)
        # Textos da tela
        self.tela_historico_label_1.setText("Histórico")
        self.tela_historico_listWidget.setSortingEnabled(False)
        __sortingEnabled = self.tela_historico_listWidget.isSortingEnabled()
        self.tela_historico_listWidget.setSortingEnabled(False)
        self.tela_historico_listWidget.setSortingEnabled(__sortingEnabled)
        self.tela_historico_textBrowser.setHtml("")
        self.tela_historico_botao_limpar_selecao.setText("Limpar Seleção")
        self.tela_historico_botao_exportar.setText("Exportar")
        self.tela_historico_botao_importar.setText("Importar")
        self.tela_historico_botao_voltar.setText("Voltar")

        # =============================================================================================================
        # Acerta os slots entre os elementos da interface gráfica e as funções de tratamento correspondente
        # =============================================================================================================
        QtCore.QMetaObject.connectSlotsByName(Form)
        self.tela_inicial_botao_reconhecer.clicked.connect(self.mudar_tela_reconhecer)
        self.tela_inicial_botao_historico.clicked.connect(self.mudar_tela_historico)
        self.tela_inicial_botao_sobre.clicked.connect(self.mudar_tela_sobre)
        self.tela_sobre_botao_voltar.clicked.connect(self.mudar_tela_inicial)
        self.tela_reco_botao_voltar.clicked.connect(self.mudar_tela_inicial)
        self.tela_reco_botao_add_amostra.clicked.connect(self.dialogo_adcionar_amostra)
        self.tela_reco_botao_reconhecer.clicked.connect(self.reconhecer)
        self.tela_resultado_botao_voltar.clicked.connect(self.mudar_tela_inicial)
        self.tela_historico_botao_limpar_selecao.clicked.connect(self.limpar_selecao)
        self.tela_historico_botao_exportar.clicked.connect(self.exportar_historico)
        self.tela_historico_botao_importar.clicked.connect(self.importar_historico)
        self.tela_historico_botao_voltar.clicked.connect(self.mudar_tela_inicial)
        self.tela_historico_listWidget.itemSelectionChanged.connect(self.atualizar_tela_historico)
        self.tela_historico_listWidget.itemChanged.connect(self.atualizar_tela_historico)
        self.tela_reco_listWidget.itemDoubleClicked.connect(self.remover_amostra)

    # =================================================================================================================
    # Em seguida definimos as funções de tratamento das interações da GUI com o usuário e com o back-end
    # =================================================================================================================

    def mudar_tela_reconhecer(self):
        # Muda a tela para a tela "reconhecer"
        logger.debug("UI -> tela reconhecer")
        index = self.stackedWidget.indexOf(self.tela_reco)
        self.stackedWidget.setCurrentIndex(index)

    def mudar_tela_resultado(self):
        # Muda a tela para a tela "resultado"
        logger.debug("UI -> tela resultado")
        try:
            dict_resultado = self.historico.buscar_ultimo()
        except:
            logging.error("Erro ao buscar histórico (guy.py linha 292).")
            exit()
        string_formatada = formatar_resultado(dict_resultado)
        self.tela_resultado_text_browser.setText(string_formatada)
        pixmap = QPixmap(os.path.join(img_folder_path, 'temp.png'))
        self.tela_resultado_label_figura.setPixmap(pixmap)
        index = self.stackedWidget.indexOf(self.tela_resultado)
        self.stackedWidget.setCurrentIndex(index)

    def mudar_tela_historico(self):
        # Muda a tela para a tela "historico"
        logger.debug("UI -> tela historico")
        index = self.stackedWidget.indexOf(self.tela_historico)
        self.tela_historico_listWidget.clear()
        try:
            dict_resultados = self.historico.buscar_historico()
        except:
            logging.error("Erro ao buscar histórico (guy.py linha 309).")
            exit()
        for key in sorted(dict_resultados, key=int):
            text = "{}  {}".format(key, dict_resultados[key]['arquivo'])
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.tela_historico_listWidget.addItem(item)
        self.tela_historico_listWidget.setCurrentRow(0)
        self.atualizar_tela_historico()
        self.stackedWidget.setCurrentIndex(index)

    def atualizar_tela_historico(self):
        # Atualiza a tela historico com os itens corretos e a imagem adequada"
        # todo acertar de print para mostrar algo
        logger.debug("Atualizando a tela histórico")
        checked_items = []
        for index in range(self.tela_historico_listWidget.count()):
            if self.tela_historico_listWidget.item(index).checkState() == Qt.Checked:
                checked_items.append(self.tela_historico_listWidget.item(index).text())
        logger.debug(checked_items)
        try:
            historico = self.historico.buscar_historico()
        except:
            logging.error("Erro ao buscar histórico (guy.py linha 333).")
            exit()
        plt.figure(figsize=(4.5, 4.5))
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.axis('off')
        for i in checked_items:
            key = i.split(sep=" ")[0]
            valence = max(min(historico[key]['valence'], 1), -1)
            arousal = max(min(historico[key]['arousal'], 1), -1)
            plt.scatter(valence, arousal, s=400)         
            n = str(historico[key]['arquivo']).split()[0]
            plt.annotate(n, (valence-0.02,arousal-0.02))
        path = os.path.join(img_folder_path, "temp.png")
        plt.savefig(path, transparent=True)
        plt.close()
        background = Image.open(va_space_bg_img_path, 'r').convert("RGBA")
        foreground = Image.open(path, 'r').convert("RGBA")
        bg_w, bg_h = background.size
        img_w, img_h = foreground.size
        offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
        background.paste(foreground, offset, mask=foreground)
        background.save(path)

        pixmap2 = QPixmap(path)
        self.tela_historico_label_figura.setPixmap(pixmap2)
        if not len(self.tela_historico_listWidget.selectedItems()) == 0:
            item = self.tela_historico_listWidget.selectedItems()[0]
            key = item.text().split(sep=" ")[0]
            try:
                dict_resultados = self.historico.buscar_historico()
            except:
                logging.error("Erro ao buscar histórico (guy.py linha 363).")
                exit()
            string_formatada = formatar_resultado(dict_resultados[key])
            self.tela_historico_textBrowser.setText(string_formatada)

    def limpar_selecao(self):
        # limpa a selecao na tela historico
        logger.debug("Limpando seleção tela histórico")
        for index in range(self.tela_historico_listWidget.count()):
            self.tela_historico_listWidget.item(index).setCheckState(QtCore.Qt.Unchecked)


    def mudar_tela_sobre(self):
        # Muda a tela para a tela "sobre"
        logger.debug("UI -> tela sobre")
        index = self.stackedWidget.indexOf(self.tela_sobre)
        self.stackedWidget.setCurrentIndex(index)

    def mudar_tela_inicial(self):
        # Muda a tela para a tela "inicial"
        logger.debug("UI -> tela inicial")
        index = self.stackedWidget.indexOf(self.tela_inicial)
        self.stackedWidget.setCurrentIndex(index)

    def dialogo_adcionar_amostra(self):
        # Abre um diálogo de inserção de arquivos para adicionar amostras a lista de amostras pendentes
        logger.debug("UI -> Add amostras")
        file_filter = 'Audio File (*.wav)'
        response = QtWidgets.QFileDialog.getOpenFileNames(
            caption='Select a data file',
            directory=os.path.join(src_folder_path, '..'),
            filter=file_filter,
            initialFilter=file_filter
        )
        for arquivo in response[0]:
            to_remove = []
            for lrow in range(self.tela_reco_listWidget.count()):
                item = self.tela_reco_listWidget.item(lrow)
                if item.text() == arquivo:
                    to_remove.append(lrow)
            for row in to_remove:
                self.tela_reco_listWidget.takeItem(row)
            logger.debug("UI -> Amostra  {} adcionada a lista".format(response[0]))
            self.tela_reco_listWidget.addItem(arquivo)

    def remover_amostra(self, item):
        # Remove uma amostra da lista de amostras pendentes
        logger.debug("UI -> Remover amostra")
        self.tela_reco_listWidget.takeItem(self.tela_reco_listWidget.row(item))
        logger.debug("UI -> Amostra  removida")

    def reconhecer(self):
        # Aciona o motor do backend para reconhecer a(s) amostra(s) na lista de amostras pendentes
        logger.debug("UI -> BACKEND processar_lista(lista de amostras)")

        # Senão houver nenhuma amostra na lista de pendentes, avisar o usuário com um pop-up e voltar a tela de seleção
        if self.tela_reco_listWidget.count() == 0:
            logger.debug("Lista vazia")
            msg = QMessageBox()
            msg.setWindowTitle("CRNN-REM - Erro!")
            msg.setText("A lista está vazia! Selecione pelo menos uma amostra para prosseguir.")
            msg.setIcon(QMessageBox.Warning)
            x = msg.exec_()
            logger.debug(x)

        # Se houver alguma amostra, tratar
        else:
            # Para tratar melhor, tiramos da listWidget e colocamos numa lista comum
            lista_de_amostras = []
            while self.tela_reco_listWidget.count() > 0:
                lista_de_amostras.append(self.tela_reco_listWidget.takeItem(0).text())
            # Com a lista normal em mãos, devemos processar a lista

            # A escolha da próxima tela depende no numero de amostras
            if len(lista_de_amostras) == 1:
                # Se tiver só uma mostra, processa e vai para o resultado individual
                self.proxima_tela = self.tela_resultado
                logger.debug("UI -> RESULTADO")
            else:
                # Se tiver mais que uma mostra, vamos para o historico com todos os reconhecimentos carregados
                self.proxima_tela = self.tela_historico
                logger.debug("UI -> HISTORICO")

            # Abrimos uma janela pop-up com barras de progresso para o usuário entender que o programa não congelou,
            # mas sim está ocupado processando as amostras da lista
            self.pop_up = QtWidgets.QWidget()
            self.pop_up_ui = popup_progresso.Ui_Progresso()
            self.pop_up_ui.setupUi(self.pop_up)
            self.pop_up.show()
            self.pop_up.activateWindow()

            # Então prosseguimos para criar uma thread de processamento
            self.thread = backend.ThreadedProcessarLista(parent=None,
                                                                amostras=lista_de_amostras,
                                                                historico=self.historico)

            # Connectamos a thread a algumas funções para atualizar a GUI conforme o andamento e sinalizar o término
            self.thread.progresso_total.connect(self.pop_up_ui.update_progresso_total)
            self.thread.progresso_parcial.connect(self.pop_up_ui.update_progresso_parcial)
            self.thread.finished.connect(self.finished_rem)

            # Com tudo preparado, iniciamos a thread e aguardamos a chamada apos finalizado
            logger.info('Thread de processamento iniciada')
            self.thread.start()

    def finished_rem(self):
        # Função chamada após o término da tarefa de reconhecimento feita pelo back-end
        logger.info('Thread de processamento finalizada')
        if self.proxima_tela == self.tela_resultado:
            self.mudar_tela_resultado()
        elif self.proxima_tela == self.tela_historico:
            self.mudar_tela_historico()
        # Após torcar a tela de fundo para a correta, encerramos o pop-up
        logger.debug('Encerrando pop-op')
        self.pop_up.close()
        self.pop_up.destroy()

    def exportar_historico(self):
        # Exporta o histórico como JSON
        logger.debug('Exportando histórico')
        file_filter = 'JSON File (*.json)'
        response = QtWidgets.QFileDialog.getSaveFileName(
            caption='Select a destiantion to export files',
            directory=os.path.join(src_folder_path, '..'),
            filter=file_filter,
            initialFilter=file_filter
        )
        response = response[0]
        if not response == "":
            try:
                self.historico.exportar(response)
            except Exception as e:
                logging.error("Erro ao exportar histórico. {}".format(repr(e)))
                exit()

    def importar_historico(self):
        # Importa um histórico salvo anteriormente em formato JSON
        logger.debug('Importando histórico')
        file_filter = 'JSON File (*.json)'
        response = QtWidgets.QFileDialog.getOpenFileName(
            caption='Select a json file',
            directory=os.path.join(src_folder_path, '..'),
            filter=file_filter,
            initialFilter=file_filter
        )
        if not response[0] == "":
            try:
                self.historico.importar(response[0])
            except:
                logging.error("Erro ao importar histórico.")
                exit()
            self.mudar_tela_historico()
            logger.info('Histórico importado com sucesso')

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form(Form)
    Form.show()
    sys.exit(app.exec_())