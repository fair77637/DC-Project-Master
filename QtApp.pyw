import os
import platform
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

__version__ = "1.0.0"

class MainWindow(QMainWindow):
    
    def __init__(self, parent=None):
        super(MainWindow, self).__init(parent)
        self.image = QImage()
        self.dirty = False
        self.filename = None
        self.mirroredvertically = False
        self.mirroredhorizontally = False
        self.imageLabel = QLabel()
        self.imageLabel.setMinimumSize(200,200)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.setCentralWidget(self.imageLabel)