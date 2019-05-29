import os
import re
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from xes.qt.ui import Ui_MainWindow
from xes.analysis import Scan
import xes

class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.setupUi(self)
        self.tabifyDockWidget(self.dock_runs, self.dock_rois)
        self.tabifyDockWidget(self.dock_rois, self.dock_bgrois)
        self.dock_runs.raise_()
        self.tabifyDockWidget(self.dock_plot, self.dock_monitor2)
        self.dock_plot.raise_()
        self.takeCentralWidget() # No central widget will make docks expand
        self.threads = []
        self.proxies = []
        # Signals
        self.monitor1.sigAnalyzerRoiChanged.connect(self.plot.update_plot)
        self.show()

    def add_scan(self):
        try:
            path = str(QtGui.QFileDialog.getOpenFileName(self, 'Select "*.FIO" file')[0])
            return self._read_scan(path)

        except Exception as e:
            xes.Log.error(e)


    def _read_scan(self, path):
        matches = re.findall(r'.?\_(\d{5})\.FIO', os.path.split(path)[1])
        scan_no = matches[0]
        scan_no = '_' + scan_no + '_'   # add underscores to avoid moxing up image and scan numbers
        #img_path = os.path.join(path, 'pilatus_100k')
        img_path = os.path.split(path)[0]
        file_names = os.listdir(img_path)
        file_names = [f for f in file_names if scan_no in f and "tif" in f]
        files = sorted(list([os.path.join(img_path,f) for f in file_names]))
        files = [f for f in files if scan_no in f]
        xes.Log.debug("Reading {} ...".format(path))
        s = Scan(log_file = path, image_files = files)
        loader = QThread_Loader(s)
        self.threads.append(loader)
        callback = lambda n : self.update_image_counter(s.name, n)
        proxy = pg.SignalProxy(loader.imageLoaded,
            rateLimit=1, slot = lambda n : self.update_image_counter(s.name, n))
        self.proxies.append(proxy)
        loader.taskFinished.connect(lambda n : self.update_image_counter(s.name, (n,)))
        loader.start()
        xes.Log.debug("Scan {} loaded.".format(s))
        return s


    def update_image_counter(self, scan_name, images_loaded):
        param = self.tree_runs.invisibleRootItem().child(0).param
        child = param.child(scan_name)
        child.update_image_counter(images_loaded[0])


    @QtCore.pyqtSlot()
    def on_actionUpdate_triggered(self):
        self.plot.update_plot_manually()


    @QtCore.pyqtSlot()
    def on_actionAutoUpdate_triggered(self):
        self.plot.update_plot_manually()


    @QtCore.pyqtSlot()
    def on_actionNormalize_triggered(self):
        self.plot.update_plot_manually()


    @QtCore.pyqtSlot()
    def on_actionSubtractBackground_triggered(self):
        self.plot.update_plot_manually()


    @QtCore.pyqtSlot()
    def on_actionSingleAnalyzers_triggered(self):
        self.plot.update_plot_manually()


    @QtCore.pyqtSlot()
    def on_actionSingleScans_triggered(self):
        self.plot.update_plot_manually()


    @QtCore.pyqtSlot()
    def on_actionSingleImage_triggered(self):
        self.plot.update_plot_manually()


    @QtCore.pyqtSlot()
    def on_actionScanningType_triggered(self):
        self.plot.plot.plotItem.enableAutoRange()
        self.plot.update_plot_manually()


class QThread_Loader(QtCore.QThread):
    taskFinished = QtCore.pyqtSignal(int)
    imageLoaded = QtCore.pyqtSignal(int)

    def __init__(self, scan, *arg, **kwarg):
        self.scan = scan
        QtCore.QThread.__init__(self, *arg, **kwarg)

    def run(self):
        xes.Log.info("Loading scan {}, please wait...".format(self.scan.name))
        self.scan.read_logfile()
        self.scan.read_files(self.imageLoaded.emit)
        n = len(self.scan.images)
        self.taskFinished.emit(n)
