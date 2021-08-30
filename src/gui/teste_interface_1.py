import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow




class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.n = 0
        self.win_w = 500
        self.win_h = 300
        self.setGeometry(100, 100, self.win_w, self.win_h)
        self.setWindowTitle("Teste de GUI com QT")
        self.initUI()

    def initUI(self):
        self.label1 = QtWidgets.QLabel(self)
        self.label1.setText("Cliques: 0")
        self.label1.move(self.win_w / 2, (self.win_h / 2) + 20)
        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText("Clique em min!")
        self.b1.move(self.win_w / 2, (self.win_h / 2) - 20)
        self.b1.clicked.connect(self.clicked_b1)

    def clicked_b1(self):
        self.n += 1
        self.label1.setText("Cliques: {}".format(self.n))
        self.update()

    def update(self):
        self.label1.adjustSize()

def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())

window()