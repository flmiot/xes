from pyqtgraph.Qt import QtGui
from xes.mainwindow import MainWindow
from xes.analysis import Experiment
from xes.parameters import register_parameter_types

__version__ = 0.1

register_parameter_types()
experiment = Experiment()
app = QtGui.QApplication([])
gui = MainWindow()
