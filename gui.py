import sys
from pyqtgraph.Qt import QtCore, QtGui

from xes.widgets import XSMainWindow

if __name__ == '__main__':

    app = QtGui.QApplication([])
    app.setWindowIcon(QtGui.QIcon('icons/icon.png'))

    w = XSMainWindow()

    if len(sys.argv) > 1:
        # Schedule processing of input file
        timer = QtCore.QTimer()
        timer.singleShot(2000, lambda : w.parse_input_file(sys.argv[1]))


    w.show()
    app.exec_()
