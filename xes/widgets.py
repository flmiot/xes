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
from xes.parameters import ScanParameter

Log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class Monitor(QtGui.QWidget):

    sigAnalyzerRoiChanged = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__()
        self.setup_ui()
        self.rois = []

        self.proxy = pg.SignalProxy(self.image_view.getView().scene().sigMouseMoved,
            rateLimit=60, slot = self._update_cursor_position)

        self.proxies = []


    def setup_ui(self):
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.black)
        self.setPalette(p)
        layout = QtGui.QVBoxLayout()
        self.title = QtGui.QLabel()
        self.title.setText('<font color="white" size="2"><b>No scan selected</b>')
        self.image_view = pg.ImageView()

        self.label = QtGui.QLabel()
        self.label.setText('<font color="white">Cursor position</font>')

        # Get the colormap
        colors = cm.viridis(np.linspace(0,1,8))

        self.cm = pg.ColorMap(np.linspace(0,1,8), colors)
        self.image_view.setColorMap(self.cm)
        layout.addWidget(self.title)
        layout.addWidget(self.image_view)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def display(self, scan, sum = False):
        # Log.debug("Display scan {}. Summed: {}".format(scan, sum))
        img = scan.images
        if sum:
            img = np.sum(img, axis = 0)
            img = np.transpose(img, axes = [1,0])
            self.image_view.setImage(img)
        else:
            img = np.transpose(img, axes = [0,2,1])
            if scan.energies[0] > scan.energies[1]:
                self.image_view.setImage(img[::-1], xvals = np.array(scan.energies)[::-1])
            else:
                self.image_view.setImage(img, xvals = np.array(scan.energies))




        name = os.path.split(scan.log_file)[1]
        self.title.setText('<font color="white" size="2"><b>'+name+'</b>')


    def add_analyzer_roi(self, roi):
        vb = self.image_view.getView()
        vb.addItem(roi)
        # roi.sigRegionChangeFinished.connect(self.update_analyzer)

        proxy = pg.SignalProxy(roi.sigRegionChanged,
            rateLimit=2, delay = 0.1, slot = self.update_analyzer)

        self.proxies.append(proxy)

    def update_analyzer(self, *args):

        roi = args[0][0]

        if self.image_view.image is None:
            return

        image = self.image_view.getProcessedImage()

        # Extract image data from ROI
        axes = (self.image_view.axes['x'], self.image_view.axes['y'])

        _, coords = roi.getArrayRegion(image.view(np.ndarray),
            self.image_view.imageItem, axes, returnMappedCoords=True)

        if coords is None:
            return



        x,y = coords[0].flatten(), coords[1].flatten()
        pixels = []
        for xi,yi in zip(x,y):
            xi,yi = int(round(xi)),int(round(yi))
            pixels.append((yi,xi))

        # roi.analyzer.set_pixels(pixels)
        self.sigAnalyzerRoiChanged.emit()


    def _update_cursor_position(self, event):
        pos = event[0]
        x = self.image_view.getView().mapSceneToView(pos).x()
        y = self.image_view.getView().mapSceneToView(pos).y()
        fmt = '<font color="white">x: {:.2f} | y: {:.2f}</font>'.format(x,y)
        self.label.setText(fmt)


class SpectralPlot(QtGui.QWidget):

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__()
        self.setup_ui()
        self.ghosts = []

        self.proxy = pg.SignalProxy(self.plot.getPlotItem().scene().sigMouseMoved,
            rateLimit=60, slot = self._update_cursor_position)



    def setup_ui(self):

        buttons = dict()

        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.black)
        self.setPalette(p)

        layout = QtGui.QVBoxLayout()
        self.plot = pg.PlotWidget()
        self.plot.getPlotItem().addLegend()
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        self.plot.getAxis('bottom').setLabel('Energy', units='eV', **labelStyle)
        self.plot.getAxis('left').setLabel('Intensity', units='a.u.', **labelStyle)
        layout.addWidget(self.plot)
        self.setLayout(layout)

        tb = QtGui.QToolBar(self)
        b = QtGui.QToolButton(self)
        b.setIcon(QtGui.QIcon('icons/update.png'))
        b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        b.setToolTip("Update the energy loss plot (F5)")
        b.clicked.connect(self.update_plot_manually)
        tb.addWidget(b)
        buttons['refresh manual'] = b

        b = QtGui.QToolButton(self)
        b.setIcon(QtGui.QIcon('icons/auto-update.png'))
        b.setCheckable(True)
        b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        b.setToolTip("Enable to update the plot automatically whenever something changes")
        b.toggled.connect(self.update_plot_manually)
        tb.addWidget(b)
        buttons['refresh toggle'] = b

        b = QtGui.QToolButton(self)
        b.setIcon(QtGui.QIcon('icons/single_analyzers.png'))
        b.setCheckable(True)
        b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        b.setToolTip("Enable to plot seperate curves for analyzer signals")
        b.toggled.connect(self.update_plot_manually)
        tb.addWidget(b)
        buttons['single_analyzers'] = b

        b = QtGui.QToolButton(self)
        b.setIcon(QtGui.QIcon('icons/single_scans.png'))
        b.setCheckable(True)
        b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        b.setToolTip("Enable to plot seperate curves for individual scans")
        b.toggled.connect(self.update_plot_manually)
        tb.addWidget(b)
        buttons['single_scans'] = b

        b = QtGui.QToolButton(self)
        b.setIcon(QtGui.QIcon('icons/persistence-mode.png'))
        b.setCheckable(True)
        b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        b.setToolTip("Enable persistence mode")
        tb.addWidget(b)
        buttons['ghost'] = b

        b = QtGui.QToolButton(self)
        b.setIcon(QtGui.QIcon('icons/subtract_background_model.png'))
        b.setCheckable(True)
        b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        b.setToolTip("Subtract background models")
        b.toggled.connect(self.update_plot_manually)
        tb.addWidget(b)
        buttons['subtract_background'] = b

        b = QtGui.QToolButton(self)
        b.setIcon(QtGui.QIcon('icons/normalize.png'))
        b.setCheckable(True)
        b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        b.setToolTip("Normalize the area under each curve to 1000")
        b.toggled.connect(self.update_plot_manually)
        tb.addWidget(b)
        buttons['normalize'] = b

        self.label = QtGui.QLabel()
        self.label.setText('<font color="white">Cursor position</font>')



        #
        # controlWidget = QtGui.QWidget()
        # layout2 = QtGui.QHBoxLayout()
        # controlWidget.setLayout(layout2)
        # buttons['refresh toggle'] = QtGui.QPushButton("Auto Update")
        # buttons['refresh toggle'].setCheckable(True)
        # buttons['ghost'] = QtGui.QPushButton("Persistence mode")
        # buttons['ghost'].setCheckable(True)
        # buttons['sum'] = QtGui.QPushButton("Sum analyzers")
        # buttons['sum'].setCheckable(True)
        # buttons['refresh manual'] = QtGui.QPushButton("Update")
        # layout2.addWidget(buttons['refresh toggle'])
        # layout2.addWidget(buttons['refresh manual'])
        # layout2.addWidget(buttons['ghost'])
        # layout2.addWidget(buttons['sum'])

        layout.addWidget(tb)
        layout.addWidget(self.plot)
        layout.addWidget(self.label)

        #layout.addWidget(controlWidget)
        self.buttons = buttons


    def update_plot(self):
        if not self.buttons['refresh toggle'].isChecked():
            return
        self.update_plot_manually()



    def update_plot_manually(self):
        try:
            self.clear_plot()

            single_scans = self.buttons['single_scans'].isChecked()
            single_analyzers = self.buttons['single_analyzers'].isChecked()
            subtract_background = self.buttons['subtract_background'].isChecked()
            normalize = self.buttons['normalize'].isChecked()

            e, i, b = experiment.get_spectrum(
                single_analyzers, single_scans)

            # Plot current data:
            self._plot(e,i, b, single_analyzers, single_scans,
                subtract_background, normalize)


            # Plot ghosts
            if self.buttons['ghost'].isChecked():
                for ind, g in enumerate(self.ghosts):
                    if ind > 45:
                        break
                    self._plot(*g)
                self.ghosts.append([e,i, b, single_analyzers, single_scans])
            else:
                self.ghosts.clear()

        except Exception as e:
            fmt = 'Plot update failed: {}'.format(e)
            Log.error(fmt)


    def _plot(self, energies, intensities, backgrounds, single_analyzers = True,
        single_scans = True, subtract_background = True, normalize = False):

        sh = [len(energies.dtype.names),len(energies)]
        if not single_scans:
            sh = reversed(sh)
        colors = self._get_colors(*sh)
        all_analyzer_names = [a.name for a in experiment.analyzers]
        all_analyzer_names = list(all_analyzer_names)
        act_analyzer_names = [a.name for a in experiment.analyzers if a.active]
        act_analyzer_names = list(act_analyzer_names)

        # Plot analyzers:
        z = zip(range(len(energies)), energies, intensities, backgrounds)
        for ind0, energy, intensity, background in z:
            name = act_analyzer_names[ind0]
            label0 = name if single_analyzers else "All"

            # Plot scans
            if single_scans:
                for ind1, scan_name in enumerate(energy.dtype.names):
                    col = colors[ind1, ind0]

                    if subtract_background:
                        # Plot data - background
                        label = label0 + ' ({})*'.format(scan_name)
                        pe, pi = energy[scan_name], intensity[scan_name]
                        pb = background[scan_name]

                        sub = pi-pb
                        if normalize:
                            sub, _ = self._normalize_curve(sub)

                        self.plot.plot(pe,sub, pen=QtGui.QColor(*col), name=label)

                    else:
                        # Plot data
                        label = label0 + ' ({})'.format(scan_name)
                        pe, pi = energy[scan_name], intensity[scan_name]

                        if normalize:
                            pi, fac = self._normalize_curve(pi)
                        else:
                            fac = 1.0

                        self.plot.plot(pe,pi, pen=QtGui.QColor(*col), name=label)

                        # Plot background if not all zeros
                        pb = background[scan_name]
                        if np.any(pb):
                            pen = self._get_background_pen(col)
                            self.plot.plot(pe,pb * fac, pen=pen)


            else:
                color_index = all_analyzer_names.index(name)
                col = colors[color_index,0]

                scan_name = energy.dtype.names[0]

                if subtract_background:
                    label0 += ' ({})*'.format(scan_name)
                    pe, pi = energy[scan_name], intensity[scan_name]
                    pb = background[scan_name]

                    sub = pi-pb

                    if normalize:
                        sub, _ = self._normalize_curve(sub)

                    self.plot.plot(pe, sub, pen=QtGui.QColor(*col), name=label0)
                else:
                    label0 += ' ({})'.format(scan_name)

                    # Plot data
                    pe, pi = energy[scan_name], intensity[scan_name]

                    if normalize:
                        pi, fac = self._normalize_curve(pi)
                    else:
                        fac = 1.0

                    self.plot.plot(pe,pi, pen=QtGui.QColor(*col), name=label0)

                    # Plot background if not all zero
                    pb = background[scan_name]
                    if np.any(pb):
                        pen = self._get_background_pen(col)
                        self.plot.plot(pe,pb * fac, pen=pen)


    def _get_colors(self, no_of_scans, no_of_analyzers):
        colors = np.empty((no_of_scans, no_of_analyzers), dtype = '4i4')

        col = cm.rainbow(np.linspace(0,1.0,no_of_scans))

        for analyzer in range(no_of_analyzers):
            blend = (no_of_analyzers-analyzer) / no_of_analyzers * 55 + 200
            colors[:,analyzer] = col * blend

        return colors


    def _get_background_pen(self, color):
        pen = pg.mkPen(color=color, style=QtCore.Qt.DashLine)
        return pen


    def _update_cursor_position(self, event):
        pos = event[0]
        x = self.plot.getPlotItem().vb.mapSceneToView(pos).x()
        y = self.plot.getPlotItem().vb.mapSceneToView(pos).y()
        fmt = '<font color="white">x: {:.2f} | y: {:.2f}</font>'.format(x,y)
        self.label.setText(fmt)


    def _normalize_curve(self, i):
        """Return normalized curve and factor by which curve was scaled."""
        factor = np.abs(1 / np.sum(i)) * 1000.
        return i * factor, factor


    def clear_plot(self):
        pi = self.plot.getPlotItem()
        items = pi.listDataItems()

        for item in items:
            pi.legend.removeItem(item.name())
            pi.removeItem(item)



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
        docks['monitor'] = QtGui.QDockWidget('Monitor',self)
        docks['spectrum'] = QtGui.QDockWidget('Energy spectrum',self)
        # docks['calibrations'] = QtGui.QDockWidget('Energy calibrations',self)


        self.plot = SpectralPlot()
        docks['spectrum'].setWidget(self.plot)
        docks['spectrum'].widget().setMinimumSize(QtCore.QSize(400,300))


        self.monitor = Monitor()
        docks['monitor'].setWidget(self.monitor)
        docks['monitor'].widget().setMinimumSize(QtCore.QSize(400,300))

        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['scans'])
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['analyzer'])
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['background'])
        # self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['calibrations'])
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, docks['monitor'])
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, docks['spectrum'])

        self.scan_tree = ParameterTree(showHeader = False)
        self.scan_tree.setObjectName('ScanTree')
        par = Parameter.create(type='scanGroup', child_type='scan', gui=self)
        self.scan_tree.setParameters(par, showTop=False)
        par.sigUpdate.connect(self.scan_tree_handler)
        self.actionAddScan = QtGui.QAction(('Add new scan object'), self)
        self.actionAddScan.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        self.addAction(self.actionAddScan)
        self.actionAddScan.triggered.connect(par.addNew)
        self.par = par

        self.bgroi_tree = ParameterTree(showHeader = False)
        self.bgroi_tree.setObjectName('BackgroundRoiTree')
        par = Parameter.create(type='backgroundRoiGroup', child_type = 'backgroundRoi', gui=self)
        self.bgroi_tree.setParameters(par, showTop=False)
        par.sigUpdate.connect(self.bgroi_tree_handler)
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
        self.analyzer_tree.setObjectName('AnalyzerTree')
        par = Parameter.create(type='analyzerGroup', child_type = 'analyzer', gui=self)
        self.analyzer_tree.setParameters(par, showTop=False)
        par.sigUpdate.connect(self.analyzer_tree_handler)
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

            path = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Folder"))
            return self._read_scan(path)

        except Exception as e:
            Log.error(e)


    def action_add_scan_list(self, filenames):
        for path in filenames:
            scan = self._read_scan(path)
            self.par.addNew(scan = scan)

    def _read_scan(self, path):
        img_path = os.path.join(path, 'pilatus_100k')
        files = sorted(list([os.path.join(img_path,f) for f in os.listdir(img_path)]))

        # self.statusBar.showMessage('BUSY... Please wait.', 30000)

        Log.debug("Reading {} ...".format(path))
        # try to find logfile
        _, scan_name = os.path.split(path)
        log_file = scan_name + '.fio'
        test_path = os.path.join(path, log_file)
        if not os.path.isfile(test_path):
            # Check higher directory
            test_path = os.path.join(os.path.split(path)[0], log_file)
            if not os.path.isfile(test_path):
                return

        s = Scan(log_file = test_path, imgage_files = files)

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


    def parse_input_file(self, file):
        # test = [] #['H:/raw/alignment_00887', 'H:/raw/alignment_00888', 'H:/raw/alignment_00889', 'H:/raw/alignment_00890']

        self.statusBar.showMessage('BUSY... Processing input file.', 30000)
        with open(file, 'r') as input_file:
            data=input_file.read().replace('\n', '')

        # Divide input file into keyword blocks !SCANS, !ANALYZERS, !MODELS
        pattern = r'\!\s*([^\!]*)'
        blocks = re.findall(pattern, data)
        kw_pattern = r'{}[()](.*?)[()]'
        kp = r'{}\s*=\s*\'*\"*([^=,\t\'\"]*)'

        # Control keywords
        plot                = False
        single_analyzers    = False
        single_scans        = False
        subtract_background = False
        normalize           = False

        for b in blocks:
            if 'SCANS' in b:
                for s in re.findall(kw_pattern.format('scan'),b):
                    path = str(re.findall(kp.format('path'),s)[0])
                    b1 = str(re.findall(kp.format('include'),s)[0])
                    include = True if b1 == '1' or b1 == 'True' else False
                    elastic_scan = str(re.findall(kp.format('elastic-scan'),s)[0])
                    b2 = str(re.findall(kp.format('monitor-sum'),s)[0])
                    monitor_sum = True if b2 == '1' or b2 == 'True' else False
                    model = str(re.findall(kp.format('model'),s)[0])

                    scan = self._read_scan(path)

                    par = self.scan_tree.invisibleRootItem().child(0).param
                    par.addNew(scan = scan, elastic_scan = elastic_scan,
                        bg_model = model, include = include,
                        monitor_sum = monitor_sum)


            elif 'ANALYZERS' in b:
                for a in re.findall(kw_pattern.format('analyzer'), b):
                    posx = float(re.findall(kp.format('position-x'),a)[0])
                    posy = float(re.findall(kp.format('position-y'),a)[0])
                    height = float(re.findall(kp.format('height'),a)[0])
                    width = float(re.findall(kp.format('width'),a)[0])
                    angle = int(re.findall(kp.format('angle'),a)[0])
                    b1 = str(re.findall(kp.format('include'),a)[0])
                    include = True if b1 == '1' or b1 == 'True' else False
                    b2 =  str(re.findall(kp.format('pixel-wise'),a)[0])
                    pixel_wise = True if b2 == '1' or b2 == 'True' else False
                    e_offset = float(re.findall(kp.format('energy-offset'),a)[0])

                    par = self.analyzer_tree.invisibleRootItem().child(0).param
                    par.addNew(position = [posx,posy], size = [height, width],
                        angle = angle, include = include, pixel_wise = pixel_wise,
                        e_offset = e_offset)

            elif 'MODELS' in b:
                for m in re.findall(kw_pattern.format('model'), b):
                    refdata = str(re.findall(kp.format('reference-data'),m)[0])
                    win_start = float(re.findall(kp.format('window-start'),m)[0])
                    win_end = float(re.findall(kp.format('window-end'),m)[0])
                    offset = float(re.findall(kp.format('vertical-offset'),m)[0])

                    par = self.background_tree.invisibleRootItem().child(0).param
                    par.addNew(ref_data_name = refdata, win_start = win_start,
                        win_end = win_end, vertical_offset = offset)

            if 'PLOT' in b:
                plot = True

            else:
                # Not a recognized keyword
                fmt = "Warning! Unrecognized block in the input-file: {}"
                Log.error(fmt.format(b))
                continue

        else:
            # Finally! Fiddle it all together

            # Display first loaded scan
            par = self.scan_tree.invisibleRootItem().child(0).param
            self.monitor.display(par.children()[0].scan, sum = True)

            # Stir the analyzers
            par = self.analyzer_tree.invisibleRootItem().child(0).param
            for child in par.children():
                self.monitor.update_analyzer(((child.roi,)))
                # child.roi.sigRegionChanged.emit(child.roi)

            # Fit models
            par = self.background_tree.invisibleRootItem().child(0).param
            for child in par.children():
                child.fit_model()


            self.statusBar.showMessage('', 0)




    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_F5:
            self.plot.update_plot_manually()
            event.accept()


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
