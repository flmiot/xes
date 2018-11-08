import os
import re
#import pdb
import sys
import logging
import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, parameterTypes
from pyqtgraph.parametertree import registerParameterType

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from xes.analysis import Experiment, Analyzer, Scan, BGModel

Log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class CustomParameterItem(parameterTypes.GroupParameterItem):
    def __init__(self, *args, **kwargs):
        self.toplevel_color = QtGui.QColor(214, 239, 255)
        super(self.__class__, self).__init__(*args, **kwargs)


    def selected(self, sel):
        if sel:
            self.param.sigItemSelected.emit(self.param)

    def updateDepth(self, depth):
        ## Change item's appearance based on its depth in the tree
        ## This allows highest-level groups to be displayed more prominently.
        if depth == 1:
            for c in [0,1]:
                self.setBackground(c, QtGui.QBrush(QtGui.QColor(236,238,236)))
                self.setForeground(c, QtGui.QBrush(QtGui.QColor(0,0,0)))
                font = self.font(c)
                font.setBold(True)
                font.setPointSize(font.pointSize()+1)
                self.setFont(c, font)
                self.setSizeHint(0, QtCore.QSize(0, 25))
        else:
            for c in [0,1]:
                self.setBackground(c, QtGui.QBrush(QtGui.QColor(220,220,220)))
                self.setForeground(c, QtGui.QBrush(QtGui.QColor(50,50,50)))
                font = self.font(c)
                font.setBold(True)
                #font.setPointSize(font.pointSize()+1)
                self.setFont(c, font)
                self.setSizeHint(0, QtCore.QSize(0, 20))


class CustomParameter(parameterTypes.GroupParameter):
    itemClass       = CustomParameterItem
    sigItemSelected = QtCore.Signal(object)
    sigUpdate       = QtCore.Signal(object)


    def __init__(self, **opts):
        parameterTypes.GroupParameter.__init__(self, **opts)
        self.connect_children()


    def connect_children(self):
        for child in self.children():
            child.sigValueChanged.connect(self.update)


    def update(self, parameter):
        self.sigUpdate.emit(parameter)


    def update_lists(self, d):
        for list_name, list_content in zip(d.keys(), d.values()):
            c = self.child(list_name)
            value = c.value()
            c.setLimits(list_content)
            c.opts['value'] = value
            c.sigValueChanged.emit(c, value) # c.setValue(...) does not work fsr


class AnalyzerROI(pg.ROI):
    def __init__(self, name, position, size, angle, monitor):

        pg.ROI.__init__(self, pos=position, size=size) #, scaleSnap=True, translateSnap=True)
        self.addScaleHandle([1, 1], [0, 0])
        self.addRotateHandle([0, 0], [0.5, 0.5])
        self.setAngle(angle)

        self.analyzer = Analyzer(name)
        experiment.add_analyzer(self.analyzer)
        self.setToolTip(self.analyzer.name)
        monitor.add_analyzer_roi(self)


class AnalyzerParameter(CustomParameter):
    def __init__(self, **opts):
        self.roi = AnalyzerROI(opts['name'], opts['position'], opts['size'],
            opts['angle'], opts['gui'].monitor)

        children = []
        children.append({'name': 'Include', 'type':'bool', 'value':opts['include']})
        children.append({'name': 'Pixel-wise', 'type':'bool', 'value':opts['pixel_wise']})
        children.append({'name': 'Energy offset', 'type':'float', 'value':opts['offset'],
            'step':0.1, 'siPrefix':False, 'suffix': 'eV'})
        opts['children'] = children
        super(self.__class__, self).__init__(**opts)


    def update(self, parameter):
        self.roi.analyzer.active = self.child('Include').value()
        self.roi.analyzer.energy_offset = self.child('Energy offset').value()
        self.roi.analyzer.pixel_wise = self.child('Pixel-wise').value()
        super(self.__class__, self).update(parameter)

registerParameterType('analyzer', AnalyzerParameter, override=True)


class ScanParameter(CustomParameter):
    def __init__(self, **opts):
        scan = opts['scan']
        elastic_scan_name = opts['elastic_scan_name']
        bg_model_name = opts['bg_model_name']

        include = opts['include']
        monitor_sum = opts['monitor_sum']

        experiment.add_scan(scan, elastic_scan_name)
        names = list([s.name for s in experiment.scans])
        if elastic_scan_name not in names:
            names.append(elastic_scan_name)
        bnames = ['None']
        bnames.extend(list([b.name for b in experiment.bg_models]))
        if bg_model_name not in bnames:
            bnames.append(bg_model_name)


        c = []
        c.append({'name': 'Include', 'type':'bool', 'value':include})
        c.append({'name': 'Elastic scan', 'type':'list', 'values':names,
            'value':elastic_scan_name})
        c.append({'name': 'Monitor: SUM', 'type':'bool', 'value':monitor_sum})
        c.append({'name': 'Images', 'type':'int', 'value':0,
            'readonly': True})
        c.append({'name': 'Background model', 'type':'list',
            'values':bnames, 'value':bg_model_name})
        opts['children'] = c
        self.scan = scan
        super(self.__class__, self).__init__(**opts)


    def update_image_counter(self, images_loaded):
        self.child('Images').setValue(images_loaded)


    def update(self, parameter):
        self.scan.active =  self.child('Include').value()
        elastic_scan_name = self.child('Elastic scan').value()
        experiment.change_elastic_scan(self.scan, elastic_scan_name)


        # Set background model
        bg_name = self.child('Background model').value()

        if bg_name != 'None':
            try:
                bg_model_names = list([b.name for b in experiment.bg_models])
                index = bg_model_names.index(bg_name)
                bg_model = experiment.bg_models[index]
                self.scan.set_background_model(bg_model)
            except:
                Log.debug("Background model could not be assigned.")

        super(self.__class__, self).update(parameter)


    def update_lists(self):
        d = dict()
        d['Elastic scan'] = list([s.name for s in experiment.scans])
        l = ['None']
        l.extend(list([b.name for b in experiment.bg_models]))
        d['Background model'] = l
        super(self.__class__, self).update_lists(d)

registerParameterType('scan', ScanParameter, override=True)


class BGModelParameter(CustomParameter):
    def __init__(self, **opts):
        self.model = BGModel()
        self.model.name = opts['name']
        experiment.add_bg_model(self.model)
        names = ['None']
        names.extend(list([scan.name for scan in experiment.scans]))
        if opts['reference'] not in names:
            names.append(opts['reference'])

        c = []
        #c.append({'name': 'Fit', 'type':'action'})
        c.append({'name': 'Reference data', 'type':'list', 'values':names,
            'value':opts['reference']})
        c.append({'name': 'Window (start)', 'type':'float',
            'value': opts['win_start'], 'siPrefix':False, 'suffix': 'eV'})
        c.append({'name': 'Window (end)', 'type':'float',
            'value': opts['win_end'], 'siPrefix':False, 'suffix': 'eV'})
        c.append({'name': 'Vertical offset', 'type':'float',
            'value':opts['vertical_offset']})
        # c.append({'name': 'Fitting window', 'type':'str', 'value':'0,0'})
        opts['children'] = c
        super(self.__class__, self).__init__(**opts)
        #self.child('Fit').sigActivated.connect(self.fit_model)


    def update(self, parameter):
        self.fit_model()
        CustomParameter.update(self, parameter)


    def fit_model(self):
        # try:
        ref_name = self.child('Reference data').value()
        if ref_name is 'None':
            raise Exception('No reference data selected.')

        names = list([s.name for s in experiment.scans])
        scan = experiment.scans[names.index(ref_name)]
        elastic = experiment.elastic_scans[names.index(ref_name)]
        self.model.set_data(scan, elastic)

        offset = self.child('Vertical offset').value()
        self.model.set_voffset(offset)
        #
        # window_str = self.child('Fitting window').value()
        # pattern = r'([-+]?\d*\.\d+|[-+]?\d+)'
        # numbers = re.findall(pattern, window_str)
        # window = list([float(n) for n in numbers])

        wstart = float(self.child('Window (start)').value())
        wend = float(self.child('Window (end)').value())

        self.model.set_window([wstart, wend])

        self.model.fit(experiment.analyzers, method = 'pearson7')

        # except Exception as e:
        #     Log.error('Model fitting failed: {}'.format(e))


    def update_lists(self):
        d = dict()
        d['Reference data'] = list([s.name for s in experiment.scans])
        super(self.__class__, self).update_lists(d)

registerParameterType('bgModel', BGModelParameter, override=True)


class CustomGroupParameter(parameterTypes.GroupParameter):

    sigUpdate = QtCore.Signal(object)
    sigItemSelected = QtCore.Signal(object)

    def __init__(self, **opts):
        opts['name'] = 'params'
        parameterTypes.GroupParameter.__init__(self, **opts)



    def addNew(self, typ=None, **opts):
        try:
            opts['type'] = self.opts['child_type']
            par = Parameter.create(**opts)
            self.addChild(par)
            self.sigUpdate.emit(par)
            self.connect_children()
        except Exception as e:
            Log.error("New entry could not be added: {}".format(e), exc_info = True)


    def update_lists(self):
        for child in self.children():
            child.update_lists()


    def connect_children(self):
        for child in self.children():
            child.sigUpdate.connect(self.update)
            child.sigItemSelected.connect(self.update)


    def update(self, parameter):
        self.sigUpdate.emit(parameter)


class AnalyzerGroupParameter(CustomGroupParameter):
    def __init__(self, **opts):
        opts['addText'] = 'Add analyzer'
        super(self.__class__, self).__init__(**opts)


    def addNew(self, position = [128,128], size = [20,20], angle = 0.0,
        include = True, pixel_wise = False, e_offset = 0.0):
        opts = {}
        opts['name'] = 'Analyzer {}'.format( len(experiment.analyzers))
        opts['gui'] = self.opts['gui']
        opts['position'] = position
        opts['size'] = size
        opts['angle'] = angle
        opts['include'] = include
        opts['pixel_wise'] = pixel_wise
        opts['offset'] = e_offset
        super(self.__class__, self).addNew(**opts)
        # name=name, type=self.opts['child_type'], gui=self.opts['gui']

registerParameterType('analyzerGroup', AnalyzerGroupParameter, override=True)


class ScanGroupParameter(CustomGroupParameter):
    def __init__(self, **opts):
        opts['addText'] = 'Add scan'
        super(self.__class__, self).__init__(**opts)


    def addNew(self, typ=None, scan = None, elastic_scan = None,
        bg_model = 'None', include = True, monitor_sum = True):
        """
        Will switch to interactive mode and ask for a scan to open if no scan is
        provided (i.e. scan = None).
        """

        if scan is None:
            scan = self.opts['gui'].action_read_scan()

        opts = {}
        opts['scan'] = scan
        opts['name'] = scan.name
        opts['include'] = include
        opts['monitor_sum'] = monitor_sum

        if elastic_scan is not None:
            opts['elastic_scan_name'] = elastic_scan
        else:
            opts['elastic_scan_name'] = scan.name

        opts['bg_model_name'] = bg_model

        super(self.__class__, self).addNew(**opts)
        self.update_lists()


registerParameterType('scanGroup', ScanGroupParameter, override=True)


class BGModelGroupParameter(CustomGroupParameter):
    def __init__(self, **opts):
        opts['addText'] = 'Add model'
        super(self.__class__, self).__init__(**opts)

    def addNew(self, ref_data_name = 'None', win_start = 0.0, win_end = 0.0,
        vertical_offset = 0.0):
        opts = {}
        opts['name'] = 'Model {}'.format(len(experiment.bg_models) + 1)
        opts['reference'] = ref_data_name
        opts['win_start'] = win_start
        opts['win_end'] = win_end
        opts['vertical_offset'] = vertical_offset

        super(self.__class__, self).addNew(**opts)




registerParameterType('bgModelGroup', BGModelGroupParameter, override=True)







# class ScanParameterItem(parameterTypes.GroupParameterItem):
#     def __init__(self, *args, **kwargs):
#         super(self.__class__, self).__init__(*args, **kwargs)
#         # for c in [0,1]:
#         #     self.setBackground(c, QtGui.QBrush(QtGui.QColor(255,100,100)))
#         #     self.setForeground(c, QtGui.QBrush(QtGui.QColor(220,220,255)))
#         #     font = self.font(c)
#         #     font.setBold(True)
#         #     font.setPointSize(font.pointSize()+1)
#         #     self.setFont(c, font)
#         #     self.setSizeHint(0, QtCore.QSize(0, 25))
#
#
#     def selected(self, sel):
#         if sel:
#             self.param.sigScanSelected.emit()

# class ScanParameter(Parameter):
#
#     itemClass = ScanParameterItem
#     sigScanSelected = QtCore.Signal()
#     sigUpdatePlots = QtCore.Signal()
#
#
#     def __init__(self, scan, monitor, **opts):
#         super(self.__class__, self).__init__(**opts)
#         self.scan = scan
#         self.monitor = monitor
#         self.sigScanSelected.connect(self.display)
#
#         try:
#             param = self.child('Monitor: SUM')
#             param.sigValueChanged.connect(self.display)
#
#             param = self.child('Include')
#             param.sigValueChanged.connect(self.update_scan)
#             param.sigValueChanged.connect(self.sigUpdatePlots.emit)
#
#             param = self.child('Elastic scan')
#             param.sigValueChanged.connect(self.update_scan)
#
#             param = self.child('Background model')
#             param.sigValueChanged.connect(self.update_scan)
#
#
#         except:
#             pass
#
#     def display(self):
#         summed =  self.child('Monitor: SUM').value()
#         self.monitor.display(self.scan, sum = summed)
#
#     def update_scan(self):
#         self.scan.active =  self.child('Include').value()
#         elastic_scan_name = self.child('Elastic scan').value()
#         experiment.change_elastic_scan(self.scan, elastic_scan_name)
#
#         bg_string = self.child('Background model').value()
#         if bg_string is not 'None':
#             bg_names = list([b.name for b in experiment.bg_models])
#             self.scan.bg_model = experiment.bg_models[bg_names.index(bg_string)]
#
#
# class ScanGroupParameter(parameterTypes.GroupParameter):
#
#     sigUpdatePlots = QtCore.Signal()
#     sigScanAdded = QtCore.Signal()
#
#     def __init__(self, gui, **opts):
#         opts['type'] = 'group'
#         opts['addText'] = "Add scan"
#         parameterTypes.GroupParameter.__init__(self, **opts)
#         self.gui = gui
#
#     def addNew(self):
#         # try:
#         new_scan = self.gui.action_read_scan()
#         experiment.add_scan(new_scan, new_scan)
#
#         scans = experiment.scans
#         elastic_scans = experiment.elastic_scans
#         bg_models = experiment.bg_models
#         names = list([scan.name for scan in scans])
#         bg_model_names = ['None']
#         bg_model_names.extend(list([b.name for b in experiment.bg_models]))
#
#         p = {'name': new_scan.name, 'type':'scan', 'monitor':self.gui.monitor,
#             'scan' : new_scan}
#         children = []
#
#         s = {'name': 'Include', 'type':'bool', 'value':new_scan.active}
#         e = {'name': 'Elastic scan', 'type':'list', 'values':names,
#             'value':new_scan.name}
#         c = {'name': 'Monitor: SUM', 'type':'bool', 'value':True}
#         l = {'name': 'Images', 'type':'int', 'value':len(new_scan.files),
#             'readonly': True}
#         b = {'name': 'Background model', 'type':'list',
#             'values':bg_model_names, 'value':'None'}
#
#         children.extend([s,e,l,c,b])
#         p['children'] = children
#
#         par = Parameter.create(**p)
#         par.sigUpdatePlots.connect(self.gui.plot.update_plot)
#         self.addChild(par)
#
#         # Update lists of all other children
#         self.update_lists()
#         #parameterTypes.GroupParameter.addNew()
#         # except Exception as e:
#         #     print(e)
#         self.sigScanAdded.emit()
#
#     def update_lists(self):
#         scans = experiment.scans
#         names = list([scan.name for scan in scans])
#         bg = experiment.bg_models
#         bg_names = list([b.name for b in bg])
#
#         for child in self.children():
#             child.child('Elastic scan').setLimits(names)
#             child.child('Background model').setLimits(bg_names)
#
#
# class BackgroundParameter(parameterTypes.GroupParameter):
#
#
#
#     def __init__(self, **opts):
#         self.model = BGModel()
#         experiment.bg_models.append(self.model)
#
#         children = []
#         ac = {'name': 'Fit', 'type':'action'}
#
#         names = ['None']
#         names.extend(list([scan.name for scan in experiment.scans]))
#
#         r = {'name': 'Reference data', 'type':'list', 'values':names,
#             'value':'None'}
#         fw = {'name': 'Fitting window', 'type':'str', 'value':'0,0'}
#         children.extend([ac, r, fw])
#         opts['children'] = children
#         parameterTypes.GroupParameter.__init__(self, **opts)
#         self.child('Fit').sigActivated.connect(self.fit_model)
#
#
#     def fit_model(self):
#         # try:
#         ref_name = self.child('Reference data').value()
#         names = list([s.name for s in experiment.scans])
#         scan = experiment.scans[names.index(ref_name)]
#         self.model.set_data(scan)
#
#         window_str = self.child('Fitting window').value()
#         pattern = r'([-+]?\d*\.\d+|[-+]?\d+)'
#         numbers = re.findall(pattern, window_str)
#         window = list([float(n) for n in numbers])
#         self.model.set_window(window)
#
#         self.model.fit(experiment.analyzers, method = 'pearson7')
#         # except Exception as e:
#         #     Log.error('Model fitting failed: {}'.format(e))
#
#
#
# class BackgroundGroupParameter(parameterTypes.GroupParameter):
#
#     sigModelAdded = QtCore.Signal()
#
#     def __init__(self, gui, **opts):
#         opts['addText'] = 'Add background model'
#         parameterTypes.GroupParameter.__init__(self, **opts)
#         self.gui = gui
#
#     def addNew(self):
#         name = 'Model {}'.format(len(self.children()) + 1)
#         p = {'name': name, 'type':'background'}
#         par = Parameter.create(**p)
#         self.addChild(par)
#
#         self.sigModelAdded.emit()
#
#     def update_lists(self):
#         scans = experiment.scans
#         names = list([s.name for s in scans])
#
#         for child in self.children():
#             child.child('Reference data').setLimits(names)
#
#
#

#
#
# class AnalyzerParameter(parameterTypes.SimpleParameter):
#
#     sigUpdatePlots = QtCore.Signal()
#
#     def __init__(self, **opts):
#         opts['type'] = 'bool'
#         super(self.__class__, self).__init__(**opts)
#         self.roi = AnalyzerROI(20, self.opts['name'])
#         opts['monitor'].add_analyzer_roi(self.roi)
#
#         self.sigValueChanged.connect(self.update_analyzer)
#
#     def update_analyzer(self):
#         self.roi.analyzer.active = self.value()
#         self.sigUpdatePlots.emit()
#
#
#
# class AnalyzerGroupParameter(parameterTypes.GroupParameter):
#
#     def __init__(self, monitor, **opts):
#         opts['type'] = 'group'
#         opts['addText'] = "Add analyzer"
#         parameterTypes.GroupParameter.__init__(self, **opts)
#         self.monitor = monitor
#
#     def addNew(self):
#         l = len(experiment.analyzers)
#         name = 'Analyzer {}'.format(l)
#         p = {'name':name, 'type':'analyzer', 'value':True,
#             'monitor':self.monitor}
#         self.addChild(p)
#
#
#
# registerParameterType('scan', ScanParameter, override=True)
# registerParameterType('scanGroup', ScanGroupParameter, override=True)
# registerParameterType('background', BackgroundParameter, override=True)
# registerParameterType('backgroundGroup', BackgroundGroupParameter, override=True)
# registerParameterType('analyzer', AnalyzerParameter, override=True)
# registerParameterType('analyzerGroup', AnalyzerGroupParameter, override=True)


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
        print("yoooo")

        roi = args[0][0]

        if self.image_view.image is None:
            print("noooo")
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

        roi.analyzer.set_pixels(pixels)
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

            e, i, b = experiment.get_energy_loss_spectrum(
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
        docks['background'] = QtGui.QDockWidget("Background models", self)
        docks['monitor'] = QtGui.QDockWidget('Monitor',self)
        docks['spectrum'] = QtGui.QDockWidget('Energy loss spectrum',self)


        self.plot = SpectralPlot()
        docks['spectrum'].setWidget(self.plot)
        docks['spectrum'].widget().setMinimumSize(QtCore.QSize(400,300))


        self.monitor = Monitor()
        docks['monitor'].setWidget(self.monitor)
        docks['monitor'].widget().setMinimumSize(QtCore.QSize(400,300))

        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['scans'])
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['analyzer'])
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, docks['background'])
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


        self.background_tree = ParameterTree(showHeader = False)
        self.background_tree.setObjectName('BGModelTree')
        par = Parameter.create(type='bgModelGroup', child_type = 'bgModel', gui=self)
        self.background_tree.setParameters(par, showTop=False)
        par.sigUpdate.connect(self.bgmodel_tree_handler)
        self.actionAddModel = QtGui.QAction(('Add new background model'), self)
        self.actionAddModel.setShortcut(QtGui.QKeySequence("Ctrl+B"))
        self.addAction(self.actionAddModel)
        self.actionAddModel.triggered.connect(par.addNew)

        self.analyzer_tree = ParameterTree(showHeader = False)
        self.analyzer_tree.setObjectName('AnalyzerTree')
        par = Parameter.create(type='analyzerGroup', child_type = 'analyzer', gui=self)
        self.analyzer_tree.setParameters(par, showTop=False)
        par.sigUpdate.connect(self.analyzer_tree_handler)
        self.actionAddAnalyzer = QtGui.QAction(('Add new analyzer'), self)
        self.actionAddAnalyzer.setShortcut(QtGui.QKeySequence("Ctrl+A"))
        self.addAction(self.actionAddAnalyzer)
        self.actionAddAnalyzer.triggered.connect(par.addNew)

        #
        # add_scan = QtGui.QAction("&Add scan", self)
        # add_scan.setShortcut("Ctrl+A")
        # add_scan.setStatusTip('Add a scan by selecting a folder')
        # add_scan.triggered.connect(self.action_add_scan)



        docks['scans'].setWidget(self.scan_tree)
        # docks['scans'].widget().setMinimumSize(QtCore.QSize(400,300))
        docks['background'].setWidget(self.background_tree)
        # docks['background'].widget().setMinimumSize(QtCore.QSize(400,300))
        docks['analyzer'].setWidget(self.analyzer_tree)
        # docks['analyzer'].widget().setMinimumSize(QtCore.QSize(400,300))

        # # Final touches
        self.setWindowTitle('XRS analysis (GUI)')
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
            param = self.background_tree.invisibleRootItem().child(0).param
            param.update_lists()

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

            if parameter.name() == 'Pixel-wise':
                self.plot.update_plot()

            if parameter.name() == 'Energy offset':
                self.plot.update_plot()


    def bgmodel_tree_handler(self, parameter):

        # if isinstance(parameter, )
        param = self.scan_tree.invisibleRootItem().child(0).param
        param.update_lists()

        self.plot.update_plot()

    # def build_scan_tree(self):

        # scans = experiment.scans
        # elastic_scans = experiment.elastic_scans
        #
        # all_scan_names = list([os.path.splitext(os.path.split(scan.log_file)[1])[0] for scan in scans])
        # parameters = []
        # for ind, scan in enumerate(scans):
        #     name = os.path.splitext(os.path.split(scan.log_file)[1])[0]
        #     p = {'name': name, 'type':'scan', 'monitor':self.monitor, 'scan' : scan}
        #     children = []
        #     s = {'name': 'Include', 'type':'bool', 'value':scan.active}
        #     e = {'name': 'Elastic scan', 'type':'list', 'values':all_scan_names,
        #         'value':elastic_scans[ind]}
        #     c = {'name': 'Monitor: SUM', 'type':'bool', 'value':True}
        #     l = {'name': 'Images', 'type':'int', 'value':len(scan.files), 'readonly': True}
        #     children.extend([s,e,l,c])
        #     p['children'] = children
        #     parameters.append(p)





    #
    # def build_background_tree(self):
    #
    #     # scans = experiment.scans
    #     # elastic_scans = experiment.elastic_scans
    #     #
    #     # all_scan_names = list([os.path.splitext(os.path.split(scan.log_file)[1])[0] for scan in scans])
    #     # parameters = []
    #     # for ind, scan in enumerate(scans):
    #     #     name = os.path.splitext(os.path.split(scan.log_file)[1])[0]
    #     #     p = {'name': name, 'type':'scan', 'monitor':self.monitor, 'scan' : scan}
    #     #     children = []
    #     #     s = {'name': 'Include', 'type':'bool', 'value':scan.active}
    #     #     e = {'name': 'Elastic scan', 'type':'list', 'values':all_scan_names,
    #     #         'value':elastic_scans[ind]}
    #     #     c = {'name': 'Monitor: SUM', 'type':'bool', 'value':True}
    #     #     l = {'name': 'Images', 'type':'int', 'value':len(scan.files), 'readonly': True}
    #     #     children.extend([s,e,l,c])
    #     #     p['children'] = children
    #     #     parameters.append(p)
    #
    #     par = Parameter.create(name='params', type='backgroundGroup', children=[],
    #         gui=self)
    #     self.background_tree.setParameters(par, showTop=False)

    #
    # def build_analyzer_tree(self):
    #
    #     analyzers = experiment.analyzers
    #
    #     parameters = []
    #     for ind, analyzer in enumerate(analyzers):
    #         name = "Analyzer {}".format(ind)
    #         p = {'name': name, 'type':'analyzer', 'value':False, 'analyzer':analyzer}
    #         parameters.append(p)
    #     p = {'name':'params', 'type':'analyzerGroup', 'children':parameters,
    #         'monitor': self.monitor}
    #     par = Parameter.create(**p)
    #
    #     for child in par.children():
    #         child.sigUpdatePlots.connect(self.plot.update_plot)
    #     p = {'name':'params', 'type':'analyzerGroup', 'children':parameters,
    #         'monitor': self.monitor}
    #     par = Parameter.create(**p)
    #     self.analyzer_tree.setParameters(par, showTop=False)


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
        img_path = os.path.join(path, 'lmbd')
        files = sorted(list([os.path.join(img_path,f) for f in os.listdir(img_path)]))

        # self.statusBar.showMessage('BUSY... Please wait.', 30000)


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





if __name__ == '__main__':

    # folder = sys.argv[1]
    # scans = list([s for s in sys.argv[2:]])
    # for ind, s in enumerate(scans):
    #     scans[ind] = os.path.join(folder, s)
    #
    #

    experiment = Experiment()



    app = QtGui.QApplication([])
    app.setWindowIcon(QtGui.QIcon('icons/icon.png'))

    w = XSMainWindow()


    # try:
    # msg = QtGui.QMessageBox()
    # msg.setIcon(QtGui.QMessageBox.Information)
    # msg.setText("This is a message box")
    # # msg.setInformativeText("This is additional information")
    # # msg.setWindowTitle("MessageBox demo")
    # # msg.setDetailedText("The details are as follows:")
    # retval = msg.show()


    if len(sys.argv) > 1:
        # Schedule processing of input file
        timer = QtCore.QTimer()
        timer.singleShot(2000, lambda : w.parse_input_file(sys.argv[1]))


    # except Exception as e:
    #     Log.error("No valid input file was found: {}".format(e))

    w.show()
    app.exec_()
