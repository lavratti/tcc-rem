# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tela_inicial.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.n = 0
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(550, 400)
        font = QtGui.QFont()
        font.setPointSize(8)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label1 = QtWidgets.QLabel(self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(200, 200, 150, 25))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label1.setFont(font)
        self.label1.setAlignment(QtCore.Qt.AlignCenter)
        self.label1.setObjectName("label1")
        self.button1 = QtWidgets.QPushButton(self.centralwidget)
        self.button1.setGeometry(QtCore.QRect(200, 150, 150, 25))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.button1.setFont(font)
        self.button1.setObjectName("button1")
        self.button1.clicked.connect(self.clicked_b1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 528, 21))
        self.menubar.setObjectName("menubar")
        self.menuTelas = QtWidgets.QMenu(self.menubar)
        self.menuTelas.setObjectName("menuTelas")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionReconhecer = QtWidgets.QAction(MainWindow)
        self.actionReconhecer.setObjectName("actionReconhecer")
        self.actionHist_rico = QtWidgets.QAction(MainWindow)
        self.actionHist_rico.setObjectName("actionHist_rico")
        self.actionConfigura_es = QtWidgets.QAction(MainWindow)
        self.actionConfigura_es.setObjectName("actionConfigura_es")
        self.actionSobre = QtWidgets.QAction(MainWindow)
        self.actionSobre.setObjectName("actionSobre")
        self.actionTela_Inicia = QtWidgets.QAction(MainWindow)
        self.actionTela_Inicia.setObjectName("actionTela_Inicia")
        self.menuTelas.addAction(self.actionTela_Inicia)
        self.menuTelas.addSeparator()
        self.menuTelas.addAction(self.actionReconhecer)
        self.menuTelas.addAction(self.actionHist_rico)
        self.menuTelas.addAction(self.actionConfigura_es)
        self.menuTelas.addSeparator()
        self.menuTelas.addAction(self.actionSobre)
        self.menubar.addAction(self.menuTelas.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label1.setToolTip(_translate("MainWindow", "Numero de vezes que o botão foi pressionado."))
        self.label1.setText(_translate("MainWindow", "Vezes clicado: ###"))
        self.button1.setStatusTip(_translate("MainWindow", "Incrementa o contador de cliques."))
        self.button1.setText(_translate("MainWindow", "Clique em min!"))
        self.menuTelas.setTitle(_translate("MainWindow", "Inicio"))
        self.actionReconhecer.setText(_translate("MainWindow", "Reconhecer..."))
        self.actionReconhecer.setStatusTip(_translate("MainWindow", "Inicia o assistente de reconehcimento de emoções em música."))
        self.actionHist_rico.setText(_translate("MainWindow", "Histórico"))
        self.actionHist_rico.setStatusTip(_translate("MainWindow", "Visita o histórico de emoções reconhecidas em músicas pelo programa."))
        self.actionConfigura_es.setText(_translate("MainWindow", "Configurações"))
        self.actionConfigura_es.setStatusTip(_translate("MainWindow", "Acessa as configuações do programa."))
        self.actionSobre.setText(_translate("MainWindow", "Sobre"))
        self.actionSobre.setStatusTip(_translate("MainWindow", "Sobre o programa, os autores e o trabalho realizado."))
        self.actionTela_Inicia.setText(_translate("MainWindow", "Tela Inicial"))
        self.actionTela_Inicia.setStatusTip(_translate("MainWindow", "Vai para a tela inicial do programa."))

    def clicked_b1(self):
        self.n += 1
        self.label1.setText("Vezes clicado: {}".format(self.n))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

