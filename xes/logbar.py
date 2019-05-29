from pyqtgraph import QtGui

class Logbar(QtGui.QStatusBar):
    def write(self, message):
        if message != '\n':
            self.showMessage(message)

    def flush(self, *args):
        pass
