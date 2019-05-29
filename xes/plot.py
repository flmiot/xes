import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import matplotlib.cm as cm

import xes

import logging
Log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)



class Plot(QtGui.QWidget):

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.setup_ui()
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



        # tb = QtGui.QToolBar(self)
        # b = QtGui.QToolButton(self)
        # b.setIcon(QtGui.QIcon('icons/update.png'))
        # b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        # b.setToolTip("Update the energy loss plot (F5)")
        # b.clicked.connect(self.update_plot_manually)
        # tb.addWidget(b)
        # buttons['refresh manual'] = b
        #
        # b = QtGui.QToolButton(self)
        # b.setIcon(QtGui.QIcon('icons/auto-update.png'))
        # b.setCheckable(True)
        # b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        # b.setToolTip("Enable to update the plot automatically whenever something changes")
        # b.toggled.connect(self.update_plot_manually)
        # tb.addWidget(b)
        # buttons['refresh toggle'] = b
        #
        # b = QtGui.QToolButton(self)
        # b.setIcon(QtGui.QIcon('icons/scanning_type.png'))
        # b.setCheckable(True)
        # b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        # b.setToolTip("Plot scanning type spectra (HERFD)")
        # b.toggled.connect(self.switch_to_scanning_type)
        # tb.addWidget(b)
        # buttons['scanning_type'] = b
        #
        # b = QtGui.QToolButton(self)
        # b.setIcon(QtGui.QIcon('icons/single_analyzers.png'))
        # b.setCheckable(True)
        # b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        # b.setToolTip("Enable to plot seperate curves for analyzer signals")
        # b.toggled.connect(self.update_plot_manually)
        # tb.addWidget(b)
        # buttons['single_analyzers'] = b
        #
        #
        # b = QtGui.QToolButton(self)
        # b.setIcon(QtGui.QIcon('icons/single_scans.png'))
        # b.setCheckable(True)
        # b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        # b.setToolTip("Enable to plot seperate curves for individual scans")
        # b.toggled.connect(self.update_plot_manually)
        # tb.addWidget(b)
        # buttons['single_scans'] = b
        #
        #
        # b = QtGui.QToolButton(self)
        # b.setIcon(QtGui.QIcon('icons/persistence-mode.png'))
        # b.setCheckable(True)
        # b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        # b.setToolTip("Enable persistence mode")
        # b.setVisible(False)
        # tb.addWidget(b)
        # buttons['ghost'] = b
        #
        #
        # b = QtGui.QToolButton(self)
        # b.setIcon(QtGui.QIcon('icons/subtract_background_model.png'))
        # b.setCheckable(True)
        # b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        # b.setToolTip("Subtract background models")
        # b.toggled.connect(self.update_plot_manually)
        # # b.setVisible(False)
        # tb.addWidget(b)
        # buttons['subtract_background'] = b
        #
        #
        # b = QtGui.QToolButton(self)
        # b.setIcon(QtGui.QIcon('icons/normalize.png'))
        # b.setCheckable(True)
        # b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        # b.setToolTip("Normalize the area under each curve to 1000")
        # b.toggled.connect(self.update_plot_manually)
        # # b.setVisible(False)
        # tb.addWidget(b)
        # buttons['normalize'] = b
        #
        # b = QtGui.QToolButton(self)
        # b.setIcon(QtGui.QIcon('icons/normalize_before_sum_scans.png'))
        # b.setCheckable(True)
        # b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        # b.setToolTip("Normalize each scan before summation")
        # b.toggled.connect(self.update_plot_manually)
        # # b.setVisible(False)
        # tb.addWidget(b)
        # buttons['normalize_scans'] = b
        #
        # b = QtGui.QToolButton(self)
        # b.setIcon(QtGui.QIcon('icons/normalize_before_sum_analyzers.png'))
        # b.setCheckable(True)
        # b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        # b.setToolTip("Normalize each analyzer before summation")
        # b.toggled.connect(self.update_plot_manually)
        # # b.setVisible(False)
        # tb.addWidget(b)
        # buttons['normalize_analyzers'] = b
        #
        # b = QtGui.QToolButton(self)
        # b.setIcon(QtGui.QIcon('icons/single_image.png'))
        # b.setCheckable(True)
        # b.setStyleSheet("QToolButton:checked { background-color: #f4c509;}")
        # fmt = "Check to display the corresponding single image spectrum for the" +\
        #     " energy selected in the monitor window."
        # b.setToolTip(fmt)
        # b.toggled.connect(self.update_plot_manually)
        # # b.setVisible(False)
        # tb.addWidget(b)
        # buttons['single_image'] = b


        self.label = QtGui.QLabel()
        self.label.setText('<font color="black">Cursor position</font>')

        self.slices_layout = QtGui.QHBoxLayout()
        label_input = QtGui.QLabel()
        label_input.setText('<font color="black">Number of images to integrate around selected: </font>')
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

        # layout.addWidget(tb)
        layout.addWidget(self.plot)
        layout.addWidget(self.label)
        layout.addLayout(self.slices_layout)

        #layout.addWidget(controlWidget)
        self.buttons = buttons

    def update_plot(self):
        if not xes.gui.actionAutoUpdate.isChecked():
            return
        self.update_plot_manually()

    def update_plot_manually(self):
        try:
            self.clear_plot()

            single_scans = xes.gui.actionSingleScans.isChecked()
            single_analyzers = xes.gui.actionSingleAnalyzers.isChecked()
            subtract_background = xes.gui.actionSubtractBackground.isChecked()
            normalize = xes.gui.actionNormalize.isChecked()
            scanning_type = xes.gui.actionScanningType.isChecked()
            # normalize_scans = xes.gui.actionSingleScans.isChecked()
            # normalize_analyzers = xes.gui.actionSingleScans.isChecked()

            try:
                slices = int(self.slices_input.text())
            except:
                slices = 1

            if xes.gui.actionSingleImage.isChecked():
                single_image = xes.gui.monitor1.image_view.currentIndex
                self.slices_input.setEnabled(True)
            else:
                single_image = None
                self.slices_input.setDisabled(True)


            analysis_result = xes.experiment.get_spectrum()


            # Plot current data:
            self._plot(analysis_result, single_analyzers, single_scans,
                scanning_type, subtract_background, normalize, single_image,
                slices, False, False)

        # Update diagnostics plots
        # diag = xes.gui.diagnostics
        # diag.update_energy_fit(analysis_result)

        except Exception as e:
            fmt = 'Plot update failed: {}'.format(e)
            Log.error(fmt)


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
