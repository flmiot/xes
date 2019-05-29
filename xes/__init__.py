import logging
from pyqtgraph.Qt import QtGui
from xes.mainwindow import MainWindow
from xes.analysis import Experiment
from xes.parameters import register_parameter_types

__version__ = 0.1
Log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

register_parameter_types()
experiment = Experiment()
app = QtGui.QApplication([])
gui = MainWindow()

handler = logging.StreamHandler(stream = gui.statusBar)
Log.addHandler(handler)
