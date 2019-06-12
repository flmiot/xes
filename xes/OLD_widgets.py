import os
import re
import logging
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree, parameterTypes


import matplotlib.cm as cm
import matplotlib.pyplot as plt

from xes import experiment
from xes.analysis import Scan
import xes.parameters
from xes.parameters import ScanParameter, AnalyzerParameter, BackgroundParameter

Log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)





class XSMainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()


        self.plot_data = []
        self.scans = []
        self.analyzers = []
        self.threads = []
        self.proxies = []

        self.setup_ui()

        # Signals
        self.monitor.sigAnalyzerRoiChanged.connect(self.plot.update_plot)

        # self.imageview.setImage(self.image_data)
        self.resize(800, 600)


    def setup_ui(self):

        docks = dict()

        self.main_menu = self.menuBar()
        # self.file_menu = self.main_menu.addMenu('&File')
        # self.file_menu.addAction(add_scan)

        self.statusBar = QtGui.QStatusBar()
        self.setStatusBar(self.statusBar)


        docks['scans'] = QtGui.QDockWidget("Scan objects", self)
        docks['analyzer'] = QtGui.QDockWidget("Analyzers", self)
        docks['background'] = QtGui.QDockWidget("Background ROIs", self)
        docks['diagnostics'] = QtGui.QDockWidget("Diagnostics", self)
        docks['monitor'] = QtGui.QDockWidget('Monitor', self)
        docks['spectrum'] = QtGui.QDockWidget('Energy spectrum',self)
        # docks['herfd'] = QtGui.QDockWidget('HERFD',self)


        self.plot = SpectralPlot(master = self)
        docks['spectrum'].setWidget(self.plot)
        docks['spectrum'].widget().setMinimumSize(QtCore.QSize(400,300))


        self.monitor = Monitor(master = self)
        docks['monitor'].setWidget(self.monitor)
        docks['monitor'].widget().setMinimumSize(QtCore.QSize(400,300))

        self.diagnostics = DiagnosticsPlot(master = self)
        docks['diagnostics'].setWidget(self.diagnostics)
        docks['diagnostics'].widget().setMinimumSize(QtCore.QSize(400,100))

        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['scans'])
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['analyzer'])
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['background'])
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['diagnostics'])
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, docks['monitor'])
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, docks['spectrum'])
        # self.addDockWidget(QtCore.Qt.RightDockWidgetArea, docks['herfd'])

        self.scan_tree = ParameterTree(showHeader = False)
        self.scan_tree.setToolTip(ScanParameter.__doc__)
        self.scan_tree.setObjectName('ScanTree')
        par = Parameter.create(type='scanGroup', child_type='scan', gui=self)
        self.scan_tree.setParameters(par, showTop=False)
        par.sigUpdate.connect(self.scan_tree_handler)
        par.sigSnippet.connect(self.view_snippet)
        self.actionAddScan = QtGui.QAction(('Add new scan object'), self)
        self.actionAddScan.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        self.addAction(self.actionAddScan)
        self.actionAddScan.triggered.connect(par.addNew)
        self.par = par

        self.bgroi_tree = ParameterTree(showHeader = False)
        self.bgroi_tree.setToolTip(BackgroundParameter.__doc__)
        self.bgroi_tree.setObjectName('BackgroundRoiTree')
        par = Parameter.create(type='backgroundRoiGroup', child_type = 'backgroundRoi', gui=self)
        self.bgroi_tree.setParameters(par, showTop=False)
        par.sigUpdate.connect(self.bgroi_tree_handler)
        par.sigSnippet.connect(self.view_snippet)
        self.actionAddBgRoi = QtGui.QAction(('Add new background ROI'), self)
        self.actionAddBgRoi.setShortcut(QtGui.QKeySequence("Ctrl+B"))
        self.addAction(self.actionAddBgRoi)
        self.actionAddBgRoi.triggered.connect(par.addNew)


        # self.background_tree = ParameterTree(showHeader = False)
        # self.background_tree.setObjectName('BGModelTree')
        # par = Parameter.create(type='bgModelGroup', child_type = 'bgModel', gui=self)
        # self.background_tree.setParameters(par, showTop=False)
        # par.sigUpdate.connect(self.bgmodel_tree_handler)
        # self.actionAddModel = QtGui.QAction(('Add new background ROI'), self)
        # self.actionAddModel.setShortcut(QtGui.QKeySequence("Ctrl+B"))
        # self.addAction(self.actionAddModel)
        # self.actionAddModel.triggered.connect(par.addNew)

        self.analyzer_tree = ParameterTree(showHeader = False)
        self.analyzer_tree.setToolTip(AnalyzerParameter.__doc__)
        self.analyzer_tree.setObjectName('AnalyzerTree')
        par = Parameter.create(type='analyzerGroup', child_type = 'analyzer', gui=self)
        self.analyzer_tree.setParameters(par, showTop=False)
        par.sigUpdate.connect(self.analyzer_tree_handler)
        par.sigSnippet.connect(self.view_snippet)
        self.actionAddAnalyzer = QtGui.QAction(('Add new analyzer'), self)
        self.actionAddAnalyzer.setShortcut(QtGui.QKeySequence("Ctrl+A"))
        self.addAction(self.actionAddAnalyzer)
        self.actionAddAnalyzer.triggered.connect(par.addNew)

        # self.calibration_tree = ParameterTree(showHeader = False)
        # self.calibration_tree.setObjectName('CalibrationTree')
        # par = Parameter.create(type='calibrationGroup', child_type = 'calibration', gui=self)
        # self.calibration_tree.setParameters(par, showTop=False)
        # par.sigUpdate.connect(self.calibration_tree_handler)
        # self.actionAddCalibration = QtGui.QAction(('Add new energy calibration'), self)
        # self.actionAddCalibration.setShortcut(QtGui.QKeySequence("Ctrl+C"))
        # self.addAction(self.actionAddCalibration)
        # self.actionAddCalibration.triggered.connect(par.addNew)

        #
        # add_scan = QtGui.QAction("&Add scan", self)
        # add_scan.setShortcut("Ctrl+A")
        # add_scan.setStatusTip('Add a scan by selecting a folder')
        # add_scan.triggered.connect(self.action_add_scan)



        docks['scans'].setWidget(self.scan_tree)
        # docks['scans'].widget().setMinimumSize(QtCore.QSize(400,300))
        docks['background'].setWidget(self.bgroi_tree)
        # docks['background'].widget().setMinimumSize(QtCore.QSize(400,300))
        docks['analyzer'].setWidget(self.analyzer_tree)
        # docks['analyzer'].widget().setMinimumSize(QtCore.QSize(400,300))
        # docks['calibrations'].setWidget(self.calibration_tree)
        # self.hplot = SpectralPlot()
        # docks['herfd'].setWidget(self.hplot)
        # docks['herfd'].widget().setMinimumSize(QtCore.QSize(400,300))

        # # Final touches
        fmt = 'XES analysis (GUI) - XES v{}'.format(xes.__version__)
        self.setWindowTitle(fmt)
        self.docks = docks


        sshFile="styles.stylesheet"
        with open(sshFile,"r") as fh:
            self.setStyleSheet(fh.read())

    def update_image_counter(self, scan_name, images_loaded):
        param = self.scan_tree.invisibleRootItem().child(0).param
        child = param.child(scan_name)
        child.update_image_counter(images_loaded[0])


    def scan_tree_handler(self, parameter):

        #Log.debug('Tree handler: scan. Parameter: {}'.format(parameter))

        if isinstance(parameter, ScanParameter):
            summed = parameter.child('Monitor: SUM').value()
            if parameter.scan.loaded:
                self.monitor.display(parameter.scan, sum = summed)
            # param = self.background_tree.invisibleRootItem().child(0).param
            # param.update_lists()

        if isinstance(parameter, parameterTypes.SimpleParameter):
            if parameter.name() == 'Monitor: SUM':
                summed = parameter.value()
                self.monitor.display(parameter.parent().scan, sum = summed)
            if parameter.name() == 'Include':
                summed = parameter.value()
                self.plot.update_plot()


    def analyzer_tree_handler(self, parameter):
        if isinstance(parameter, parameterTypes.SimpleParameter):
            if parameter.name() == 'Include':
                self.plot.update_plot()

            # if parameter.name() == 'Pixel-wise':
            #     self.plot.update_plot()

            if parameter.name() == 'Energy offset':
                self.plot.update_plot()


    def bgroi_tree_handler(self, parameter):

        # if isinstance(parameter, )
        # param = self.scan_tree.invisibleRootItem().child(0).param
        # param.update_lists()

        self.plot.update_plot()

    def calibration_tree_handler(self, parameter):
        print(parameter)


    def action_read_scan(self):
        try:

            #path = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Folder"))
            path = str(QtGui.QFileDialog.getOpenFileName(self, 'Select "*.FIO" file')[0])
            return self._read_scan(path)

        except Exception as e:
            Log.error(e)


    def action_add_scan_list(self, filenames):
        for path in filenames:
            scan = self._read_scan(path)
            self.par.addNew(scan = scan)

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

        # self.statusBar.showMessage('BUSY... Please wait.', 30000)

        Log.debug("Reading {} ...".format(path))
        # try to find logfile
        #_, scan_name = os.path.split(path)
        #log_file = scan_name + '.fio'
        #test_path = os.path.join(path, log_file)
#        if not os.path.isfile(test_path):
#            # Check higher directory
#            test_path = os.path.join(os.path.split(path)[0], log_file)
#            if not os.path.isfile(test_path):
#                fmt = "Unable to read scan {}. Logfile (*.fio) not found!"
#                Log.error(fmt.format(path))
#                return


        s = Scan(log_file = path, image_files = files)

        loader = QThread_Loader(s)
        self.threads.append(loader)
        callback = lambda n : self.update_image_counter(s.name, n)

        proxy = pg.SignalProxy(loader.imageLoaded,
            rateLimit=1, slot = lambda n : self.update_image_counter(s.name, n))
        self.proxies.append(proxy)
        loader.taskFinished.connect(lambda n : self.update_image_counter(s.name, (n,)))
        loader.start()

        # if len(experiment.scans) > 0:
        #     default_elastic = experiment.scans[0]
        # else:
        #     default_elastic = s

        # self.statusBar.showMessage("", 1)

        Log.debug("Scan {} loaded.".format(s))
        return s

    def get_selected_image_index(self):
        """ Returns the index of the currently selected image in monitor. """
        return self.monitor.image_view.currentIndex


    def parse_input_file(self, file):
        # test = [] #['H:/raw/alignment_00887', 'H:/raw/alignment_00888', 'H:/raw/alignment_00889', 'H:/raw/alignment_00890']

        self.statusBar.showMessage('BUSY... Processing input file.', 30000)

        msgBox = QtGui.QMessageBox( QtGui.QMessageBox.Information,
            "Processing...", "Preparing", QtGui.QMessageBox.NoButton )

        # Get the layout
        l = msgBox.layout()

        # Hide the default button
        l.itemAtPosition( l.rowCount() - 1, 0 ).widget().hide()

        progress = QtGui.QProgressBar()
        progress.setMinimumSize(360,25)

        # Add the progress bar at the bottom (last row + 1) and first column with column span
        l.addWidget(progress,1, 1, 1, l.columnCount(), QtCore.Qt.AlignCenter )

        msgBox.show()

        with open(file, 'r') as input_file:
            data=input_file.read().replace('\n', '')

        # Divide input file into keyword blocks !SCANS, !ANALYZERS, !MODELS
        pattern = r'\!\s*([^\!]*)'
        blocks = re.findall(pattern, data)
        kw_pattern = r'{}[()](.*?)[()]'
        kp = r'{}\s*=\s*\'*\"*([^=,\t\'\"]*)'

        # Control keywords
        keywords = dict(
            PLOT                = [False, 'refresh manual'],
            NORMALIZE           = [False, 'normalize'],
            SINGLE_SCANS        = [False, 'single_scans'],
            SINGLE_ANALYZERS    = [False, 'single_analyzers'],
            SUBTRACT_BACKGROUND = [False, 'subtract_background'],
            AUTO                = [False, 'refresh toggle']
            )

        for b in blocks:

            # Test plotting keywords
            for word in keywords:
                if word in b:
                    keywords[word][0] = True
                    break



            if 'SCANS' in b:
                for s in re.findall(kw_pattern.format('scan'),b):
                    path = str(re.findall(kp.format('path'),s)[0])
                    b1 = str(re.findall(kp.format('include'),s)[0])
                    include = True if b1 == '1' or b1 == 'True' else False
                    elastic_scan = str(re.findall(kp.format('elastic-scan'),s)[0])
                    b2 = str(re.findall(kp.format('monitor-sum'),s)[0])
                    monitor_sum = True if b2 == '1' or b2 == 'True' else False
                    range = str(re.findall(kp.format('range'),s)[0])

                    print(range)

                    scan = self._read_scan(path)

                    par = self.scan_tree.invisibleRootItem().child(0).param
                    par.addNew(scan = scan, elastic = elastic_scan,
                        range = range, include = include,
                        monitor_sum = monitor_sum)


            elif 'ANALYZERS' in b:
                for a in re.findall(kw_pattern.format('analyzer'), b):
                    posx = float(re.findall(kp.format('position-x'),a)[0])
                    posy = float(re.findall(kp.format('position-y'),a)[0])
                    height = float(re.findall(kp.format('height'),a)[0])
                    width = float(re.findall(kp.format('width'),a)[0])
                    b1 = str(re.findall(kp.format('include'),a)[0])
                    include = True if b1 == '1' or b1 == 'True' else False

                    par = self.analyzer_tree.invisibleRootItem().child(0).param
                    par.addNew(position = [posx,posy], size = [height, width],
                        include = include)

            elif 'BACKGROUNDS' in b:
                for u in re.findall(kw_pattern.format('background'), b):
                    posx = float(re.findall(kp.format('position-x'),u)[0])
                    posy = float(re.findall(kp.format('position-y'),u)[0])
                    height = float(re.findall(kp.format('height'),u)[0])
                    width = float(re.findall(kp.format('width'),u)[0])
                    b1 = str(re.findall(kp.format('include'),u)[0])
                    include = True if b1 == '1' or b1 == 'True' else False

                    par = self.bgroi_tree.invisibleRootItem().child(0).param
                    par.addNew(position = [posx,posy], size = [height, width],
                        include = include)


            else:
                # Not a recognized keyword
                fmt = "Warning! Unrecognized block in the input-file: {}"
                Log.error(fmt.format(b))
                continue

        else:
            # Finally! Fiddle it all together
            msgBox.setText('Processing analysis')
            progress.setValue(0)

            # Display first loaded scan
            par = self.scan_tree.invisibleRootItem().child(0).param
            self.monitor.display(par.children()[0].scan, sum = True)

            # Stir the analyzers
            par = self.analyzer_tree.invisibleRootItem().child(0).param
            for child in par.children():
                self.monitor.update_analyzer(((child.roi,)))
                # child.roi.sigRegionChanged.emit(child.roi)

            # Stir background rois
            par = self.bgroi_tree.invisibleRootItem().child(0).param
            for child in par.children():
                self.monitor.update_analyzer(((child.roi,)))
                # child.roi.sigRegionChanged.emit(child.roi)


            msgBox.setText('Finishing...')
            progress.setValue(50)

            # Execute plotting keywords
            for ind, value in enumerate(keywords.values()):
                msgBox.setText('Processing analysis and plotting options [{}]'.format(value[1]))
                progress.setValue(100 * ind/len(keywords.values()))
                if value[0]:
                    self.plot.buttons[value[1]].click()


            self.statusBar.showMessage('', 0)




    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_F5:
            self.plot.update_plot_manually()
            event.accept()


    def view_snippet(self, snippet_string):
        self.dialog = SnippetViewer(snippet_string)
        self.dialog.show()


class QThread_Loader(QtCore.QThread):
    taskFinished = QtCore.pyqtSignal(int)
    imageLoaded = QtCore.pyqtSignal(int)

    def __init__(self, scan, *arg, **kwarg):
        self.scan = scan
        QtCore.QThread.__init__(self, *arg, **kwarg)

    def run(self):
        Log.info("Loading scan {}, please wait...".format(self.scan.name))
        self.scan.read_logfile()
        self.scan.read_files(self.imageLoaded.emit)
        n = len(self.scan.images)
        self.taskFinished.emit(n)


class SnippetViewer(QtGui.QDialog):
    def __init__(self, snippet_string):
        QtGui.QDialog.__init__(self)
        self.setWindowTitle("Snippet")
        layout = QtGui.QVBoxLayout()
        text_window = QtGui.QTextEdit()
        text_window.setReadOnly(True)
        text_window.insertPlainText(snippet_string)
        layout.addWidget(text_window)
        self.setLayout(layout)
