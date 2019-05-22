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

class Monitor(QtGui.QWidget):

    sigAnalyzerRoiChanged = QtCore.Signal()

    def __init__(self, *args, master = None, **kwargs):
        super(self.__class__, self).__init__()
        self.setup_ui()
        self.master = master
        self.rois = []

        self.proxy = pg.SignalProxy(self.image_view.getView().scene().sigMouseMoved,
            rateLimit=60, slot = self._update_cursor_position)

        self.proxies = []


        self.image_view.sigTimeChanged.connect(self.call_for_plot_update)


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
            x =  np.array(scan.energies)
            if len(np.unique(x)) < len(img):
                x = np.arange(len(img))

            if x[0] > x[1]:
                self.image_view.setImage(img[::-1], xvals = x[::-1])
            else:
                self.image_view.setImage(img, xvals = x)


        name = os.path.split(scan.log_file)[1]
        self.title.setText('<font color="white" size="2"><b>'+name+'</b>')

    def call_for_plot_update(self):
        if self.master.plot.buttons['single_image'].isChecked():
            self.master.plot.update_plot()


    def add_analyzer_roi(self, roi):
        vb = self.image_view.getView()
        vb.addItem(roi)
        # roi.sigRegionChangeFinished.connect(self.update_analyzer)

        proxy = pg.SignalProxy(roi.sigRegionChanged,
            rateLimit=2, delay = 0.1, slot = self.update_analyzer)

        self.proxies.append(proxy)


    def add_background_roi(self, roi):
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

        # get bounding box
        x,y = coords[0].flatten(), coords[1].flatten()
        x0, x1 = np.min(x), np.max(x)
        y0, y1 = np.min(y), np.max(y)
        bbox = list([int(i) for i in [x0,y0,x1,y1]])




        # pixels = []
        # for xi,yi in zip(x,y):
        #     xi,yi = int(round(xi)),int(round(yi))
        #     pixels.append((yi,xi))

        roi.analyzer.set_roi(bbox)
        self.sigAnalyzerRoiChanged.emit()


    def _update_cursor_position(self, event):
        pos = event[0]
        x = int(self.image_view.getView().mapSceneToView(pos).x())
        y = int(self.image_view.getView().mapSceneToView(pos).y())

        try:
            if len(self.image_view.image.shape) == 3:
                z = self.image_view.currentIndex
                i = self.image_view.image[z,x,y]
            else:
                i = self.image_view.image[x,y]
        except Exception as e:
            i = 0
        fmt = '<font color="white">x: {:6d} | y: {:6d} | intensity: {:6.0f}</font>'
        fmt = fmt.format(x,y, i)
        self.label.setText(fmt)


class SpectralPlot(QtGui.QWidget):

    def __init__(self, *args, master = None, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.master = master
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
        self.plot = pg.PlotWidget(background = 'w')
        self.plot.getPlotItem().addLegend()
        labelStyle = {'color': '#000', 'font-size': '10pt'}
        self.plot.getAxis('bottom').setLabel('Energy', units='eV', **labelStyle)
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.plot.getAxis('left').setLabel('Intensity', units='a.u.', **labelStyle)
        self.plot.getAxis('left').enableAutoSIPrefix(False)
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
        b.setIcon(QtGui.QIcon('icons/scanning_type.png'))
        b.setCheckable(True)
        b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        b.setToolTip("Plot scanning type spectra (HERFD)")
        b.toggled.connect(self.switch_to_scanning_type)
        tb.addWidget(b)
        buttons['scanning_type'] = b

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
        b.setVisible(False)
        tb.addWidget(b)
        buttons['ghost'] = b


        b = QtGui.QToolButton(self)
        b.setIcon(QtGui.QIcon('icons/subtract_background_model.png'))
        b.setCheckable(True)
        b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        b.setToolTip("Subtract background models")
        b.toggled.connect(self.update_plot_manually)
        # b.setVisible(False)
        tb.addWidget(b)
        buttons['subtract_background'] = b


        b = QtGui.QToolButton(self)
        b.setIcon(QtGui.QIcon('icons/normalize.png'))
        b.setCheckable(True)
        b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        b.setToolTip("Normalize the area under each curve to 1000")
        b.toggled.connect(self.update_plot_manually)
        # b.setVisible(False)
        tb.addWidget(b)
        buttons['normalize'] = b

        b = QtGui.QToolButton(self)
        b.setIcon(QtGui.QIcon('icons/normalize_before_sum_scans.png'))
        b.setCheckable(True)
        b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        b.setToolTip("Normalize each scan before summation")
        b.toggled.connect(self.update_plot_manually)
        # b.setVisible(False)
        tb.addWidget(b)
        buttons['normalize_scans'] = b

        b = QtGui.QToolButton(self)
        b.setIcon(QtGui.QIcon('icons/normalize_before_sum_analyzers.png'))
        b.setCheckable(True)
        b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        b.setToolTip("Normalize each analyzer before summation")
        b.toggled.connect(self.update_plot_manually)
        # b.setVisible(False)
        tb.addWidget(b)
        buttons['normalize_analyzers'] = b

        b = QtGui.QToolButton(self)
        b.setIcon(QtGui.QIcon('icons/single_image.png'))
        b.setCheckable(True)
        b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        fmt = "Check to display the corresponding single image spectrum for the" +\
            " energy selected in the monitor window."
        b.setToolTip(fmt)
        b.toggled.connect(self.update_plot_manually)
        # b.setVisible(False)
        tb.addWidget(b)
        buttons['single_image'] = b


        self.label = QtGui.QLabel()
        self.label.setText('<font color="white">Cursor position</font>')

        self.slices_layout = QtGui.QHBoxLayout()
        label_input = QtGui.QLabel()
        label_input.setText('<font color="white">Number of images to integrate around selected: </font>')
        self.slices_input = QtGui.QLineEdit("1")
        self.slices_layout.addWidget(label_input)
        self.slices_layout.addWidget(self.slices_input)
        self.slices_input.setDisabled(True)




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
        layout.addLayout(self.slices_layout)

        #layout.addWidget(controlWidget)
        self.buttons = buttons

    def switch_to_scanning_type(self):

        self.plot.plotItem.enableAutoRange()
        self.update_plot_manually()

    def update_plot(self):
        if not self.buttons['refresh toggle'].isChecked():
            return
        self.update_plot_manually()



    def update_plot_manually(self):
        # try:
        self.clear_plot()

        single_scans = self.buttons['single_scans'].isChecked()
        single_analyzers = self.buttons['single_analyzers'].isChecked()
        subtract_background = self.buttons['subtract_background'].isChecked()
        normalize = self.buttons['normalize'].isChecked()
        scanning_type = self.buttons['scanning_type'].isChecked()
        normalize_scans = self.buttons['normalize_scans'].isChecked()
        normalize_analyzers = self.buttons['normalize_analyzers'].isChecked()

        try:
            slices = int(self.slices_input.text())
        except:
            slices = 1
    
        if self.buttons['single_image'].isChecked():
            single_image = self.master.get_selected_image_index()
            self.slices_input.setEnabled(True)
        else:
            single_image = None
            self.slices_input.setDisabled(True)


        analysis_result = experiment.get_spectrum()


        # Plot current data:
        self._plot(analysis_result, single_analyzers, single_scans,
            scanning_type, subtract_background, normalize, single_image,
            slices, normalize_scans, normalize_analyzers)

        # Update diagnostics plots
        diag = self.master.diagnostics
        diag.update_energy_fit(analysis_result)


        # # Plot ghosts
        # if self.buttons['ghost'].isChecked():
        #     for ind, g in enumerate(self.ghosts):
        #         if ind > 45:
        #             break
        #         self._plot(*g)
        #     self.ghosts.append([e,i, b, single_analyzers, single_scans])
        # else:
        #     self.ghosts = []

        # except Exception as e:
        #     fmt = 'Plot update failed: {}'.format(e)
        #     Log.error(fmt)


    def _plot(self, analysis_result, single_analyzers = True, single_scans = True,
        scanning_type = False, subtract_background = True, normalize = False,
        single_image = None, slices = 1, normalize_scans = False,
        normalize_analyzers = False):

        e, i, b, l = analysis_result.get_curves(
            single_scans, single_analyzers, scanning_type, single_image, slices,
            normalize_scans, normalize_analyzers)

        pens, pens_bg = self._get_pens(e, i, b, single_analyzers, single_scans)
        # print(pens)

        # Plot scans:

        z1 = zip(range(len(e)), e, i, b, l)
        for ind_s, energy, intensity, background, label in z1:
            # Plot analyzers
            z2 = zip(range(len(energy)), energy, intensity, background, label)
            for ind_a, single_e, single_i, single_b, single_l in z2:

                if subtract_background:

                    sub = single_i - single_b

                    if normalize:
                        sub, _ = self._normalize_curve(sub)

                    self.plot.plot(single_e, sub,
                        pen = pens[ind_s, ind_a], name = single_l)
                else:

                    if normalize:
                        single_i, fac = self._normalize_curve(single_i)
                    else:
                        fac = 1.0

                    self.plot.plot(single_e, single_i, name = single_l,
                        pen = pens[ind_s, ind_a])

                    self.plot.plot(single_e, single_b * fac,
                        pen = pens_bg[ind_s, ind_a])



    def _get_pens(self, e, i, b, single_analyzers, single_scans):
        no_scans = len(e)
        no_analyzers = len(e[0])
        if single_analyzers and single_scans:
            shades = cm.gist_rainbow(np.linspace(0,1.0, no_scans))
            colors = np.tile(shades, (no_analyzers, 1, 1))
            colors = np.transpose(colors, (1,0,2))

        elif single_analyzers and not single_scans:
            shades = cm.gist_rainbow(np.linspace(0, 1.0, no_analyzers))
            colors = np.tile(shades, (1,1,1))

        elif not single_analyzers and single_scans:
            shades = cm.gist_rainbow(np.linspace(0,1.0, no_scans))
            colors = np.tile(shades, (1, 1, 1))
            colors = np.transpose(colors, (1,0,2))

        else:
            shades = cm.gist_rainbow(np.linspace(0,1.0, 1))
            colors = np.tile(shades, (1,1,1))

        pens = []
        pens_bg = []
        for ind_s, scan in enumerate(e):
            pens_scan = []
            pens_scan_bg = []
            for ind_a, analyzer in enumerate(scan):
                c = QtGui.QColor(*colors[ind_s, ind_a]*255)
                pens_scan.append(pg.mkPen(color=c, style=QtCore.Qt.SolidLine))
                pens_scan_bg.append(pg.mkPen(color=c, style=QtCore.Qt.DashLine))
            pens.append(pens_scan)
            pens_bg.append(pens_scan_bg)

        return np.array(pens), np.array(pens_bg)


    def _get_background_pen(self, color):
        pen = pg.mkPen(color=color, style=QtCore.Qt.DashLine)
        return pen


    def _update_cursor_position(self, event):
        pos = event[0]
        x = self.plot.getPlotItem().vb.mapSceneToView(pos).x()
        y = self.plot.getPlotItem().vb.mapSceneToView(pos).y()
        fmt = '<font color="white">x: {:.7f} | y: {:.7f}</font>'.format(x,y)
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



class DiagnosticsPlot(QtGui.QWidget):

    def __init__(self, *args, master = None, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.master = master
        self.setup_ui()


    def setup_ui(self):


        self.plots = {}
        self.labels = {}

        layout = QtGui.QVBoxLayout()

        # Elastic fit diagnostics
        plot = pg.PlotWidget(background = None)
        labelStyle = {'color': '#000', 'font-size': '7pt'}
        plot.getAxis('bottom').setLabel('Pixel position', units='pixel', **labelStyle)
        plot.getAxis('bottom').enableAutoSIPrefix(False)
        plot.getAxis('left').setLabel('Energy', units='eV', **labelStyle)
        plot.getAxis('left').enableAutoSIPrefix(False)
        layout.addWidget(plot)
        self.plots['elastic_fit'] = plot
        self.setLayout(layout)

        label = QtGui.QLabel()
        label.setText('<font color="black">Energy calibration fits</font>')
        self.labels['elastic_fit'] = label

        layout.addWidget(label)
        layout.addWidget(plot)



    def update_energy_fit(self, analysis_result):
        plot = self.plots['elastic_fit']
        pi = plot.getPlotItem()
        items = pi.listDataItems()
        for item in items:
            pi.removeItem(item)

        fits = analysis_result.fits

        for fit in fits:
            try:
                for x, y_data, y_fit in fit:
                    plot.plot(x, y_data, pen = None, name = 'Data', symbol='o')
                    pen = pg.mkPen(color=[0,0,255], style=QtCore.Qt.DashLine)
                    plot.plot(x, y_fit, pen = pen, name = 'Fit')
            except:
                Log.debug("Invalid energy calibration could not be plotted.")


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
