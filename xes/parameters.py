import logging

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, parameterTypes, registerParameterType

from xes import experiment
from xes.analysis import Analyzer, Calibration

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
    def __init__(self, name, position, size, monitor): #angle

        pg.ROI.__init__(self, pos=position, size=size) #, scaleSnap=True, translateSnap=True)
        self.addScaleHandle([1, 1], [0, 0])
        self.addScaleHandle([0, 0], [1, 1])
        self.addScaleHandle([0, 1], [1, 0])
        self.addScaleHandle([1, 0], [0, 1])

        self.addScaleHandle([0.5, 1], [0.5, 0])
        self.addScaleHandle([0.5, 0], [0.5, 1])

        # self.addRotateHandle([0, 0], [0.5, 0.5])

        self.analyzer = Analyzer(name)
        experiment.add_analyzer(self.analyzer)
        self.setToolTip(self.analyzer.name)
        monitor.add_analyzer_roi(self)


class AnalyzerParameter(CustomParameter):
    """ Analyzer parameter class represents an analyzer object. Parameters are

    ------------------  --------------------------------------------------
    Include             Include this object in analysis
    Energy calibration  Corresponding energy calibration used for analysis
    Energy offset       Energy offset for this analyzer, relative to calibration
    ------------------  --------------------------------------------------
    """


    def __init__(self, **opts):
        self.roi = AnalyzerROI(opts['name'], opts['position'], opts['size'],
            opts['gui'].monitor) # opts['angle'],

        c = []
        c.append({'name': 'Include', 'type':'bool', 'value':opts['include']})
        names = list([c.name for c in experiment.calibrations])
        # c.append({'name': 'Energy calibration', 'type':'list', 'values':names,
        #     'value':opts['calibration']})
        # c.append({'name': 'Energy offset', 'type':'float', 'value':opts['offset'],
        #     'step':0.1, 'siPrefix':False, 'suffix': 'eV'})
        # c.append({'name': 'Offset', 'type':'int', 'value':opts['offset'],
        #     'step':1})
        opts['children'] = c
        super(self.__class__, self).__init__(**opts)


    def update(self, parameter):
        self.roi.analyzer.active = self.child('Include').value()
        # self.roi.analyzer.energy_offset = self.child('Offset').value()
        # self.roi.analyzer.pixel_wise = self.child('Pixel-wise').value()
        super(self.__class__, self).update(parameter)


    def update_lists(self):
        d = dict()
        # d['Energy calibration'] = list([c.name for c in experiment.calibrations])
        super(self.__class__, self).update_lists(d)

registerParameterType('analyzer', AnalyzerParameter, override=True)


class BackgroundROI(pg.ROI):
    def __init__(self, name, position, size, monitor): # angle,
        # Make this a different color

        pg.ROI.__init__(self, pos=position, size=size) #, scaleSnap=True, translateSnap=True)
        self.addScaleHandle([1, 1], [0, 0])
        # self.addRotateHandle([0, 0], [0.5, 0.5])
        # self.setAngle(angle)

        self.background_roi = Analyzer(name)
        experiment.add_background_roi(self.background_roi)
        self.setToolTip(self.background_roi.name)
        #monitor.add_analyzer_roi(self)


class BackgroundParameter(CustomParameter):
    def __init__(self, **opts):
        self.roi = BackgroundROI(opts['name'], opts['position'], opts['size'],
            opts['gui'].monitor) # opts['angle'],

        c = []
        c.append({'name': 'Include', 'type':'bool', 'value':opts['include']})
        opts['children'] = c
        super(self.__class__, self).__init__(**opts)


    def update(self, parameter):
        self.roi.background_roi.active = self.child('Include').value()
        super(self.__class__, self).update(parameter)

registerParameterType('backgroundRoi', BackgroundParameter, override=True)


class ScanParameter(CustomParameter):
    """ Scan parameter class represents a scan object. Parameters are

    ------------------  --------------------------------------------------
    Include             Include this object in analysis
    Scanning type       Treat as scan (e.g. HERFD)
    Monitor: SUM        When displaying this object in monitor, sum all images
    Offset (x)          Shift all images in x-direction
    Offset (y)          Shift all images in y-direction
    ------------------  --------------------------------------------------
    """

    def __init__(self, **opts):
        scan = opts['scan']

        include = opts['include']
        scanning_type = opts['scanning_type']
        monitor_sum = opts['monitor_sum']
        offset_x = opts['offset_x']
        offset_y = opts['offset_y']


        experiment.add_scan(scan)
        names = list([s.name for s in experiment.scans])

        c = []
        c.append({'name': 'Include', 'type':'bool', 'value':include})
        c.append({'name': 'Monitor: SUM', 'type':'bool', 'value':monitor_sum})
        # c.append({'name': 'Scanning type', 'type':'bool', 'value':scanning_type})
        c.append({'name': 'Images', 'type':'int', 'value':0,
            'readonly': True})
        # c.append({'name': 'Offset (x)', 'type':'int', 'value':offset_x})
        # c.append({'name': 'Offset (y)', 'type':'int', 'value':offset_y})

        opts['children'] = c
        self.scan = scan
        super(self.__class__, self).__init__(**opts)


    def update_image_counter(self, images_loaded):
        self.child('Images').setValue(images_loaded)


    def update(self, parameter):
        self.scan.active =  self.child('Include').value()
        # self.scan.offset[0] = self.child('Offset (x)').value()
        # self.scan.offset[1] = self.child('Offset (y)').value()


        # Set background model
        # bg_name = self.child('Background model').value()
        #
        # if bg_name != 'None':
        #     try:
        #         bg_model_names = list([b.name for b in experiment.bg_models])
        #         index = bg_model_names.index(bg_name)
        #         bg_model = experiment.bg_models[index]
        #         self.scan.set_background_model(bg_model)
        #     except:
        #         Log.debug("Background model could not be assigned.")

        super(self.__class__, self).update(parameter)


    # def update_lists(self):
    #     d = dict()
    #     # d['Elastic scan'] = list([s.name for s in experiment.scans])
    #     # l = ['None']
    #     # l.extend(list([b.name for b in experiment.bg_models]))
    #     # d['Background model'] = l
    #     super(self.__class__, self).update_lists(d)

registerParameterType('scan', ScanParameter, override=True)


class CalibrationParameter(CustomParameter):
    def __init__(self, **opts):
        self.calibration = Calibration()
        self.calibration.name = opts['name']
        experiment.add_calibration(self.calibration)
        names = ['None']
        names.extend(list([scan.name for scan in experiment.scans]))
        if opts['elastic_scan'] not in names:
            names.append(opts['elastic_scan'])

        analyzers = ['None']
        if opts['main_analyzer'] not in analyzers:
            analyzers.append(opts['main_analyzer'])

        c = []
        c.append({'name': 'Elastic scan', 'type':'list', 'values':names,
            'value':opts['elastic_scan']})
        c.append({'name': 'Main analyzer', 'type':'list', 'values':analyzers,
            'value':opts['main_analyzer']})
        c.append({'name': 'First frame', 'type':'int',
            'value': opts['first_frame']})
        c.append({'name': 'last frame', 'type':'int',
            'value': opts['last_frame']})
        opts['children'] = c
        super(self.__class__, self).__init__(**opts)

registerParameterType('calibration', CalibrationParameter, override=True)


# class BGModelParameter(CustomParameter):
#     def __init__(self, **opts):
#         self.model = BGModel()
#         self.model.name = opts['name']
#         experiment.add_bg_model(self.model)
#         names = ['None']
#         names.extend(list([scan.name for scan in experiment.scans]))
#         if opts['reference'] not in names:
#             names.append(opts['reference'])
#
#         c = []
#         #c.append({'name': 'Fit', 'type':'action'})
#         c.append({'name': 'Reference data', 'type':'list', 'values':names,
#             'value':opts['reference']})
#         c.append({'name': 'Window (start)', 'type':'float',
#             'value': opts['win_start'], 'siPrefix':False, 'suffix': 'eV'})
#         c.append({'name': 'Window (end)', 'type':'float',
#             'value': opts['win_end'], 'siPrefix':False, 'suffix': 'eV'})
#         c.append({'name': 'Vertical offset', 'type':'float',
#             'value':opts['vertical_offset']})
#         # c.append({'name': 'Fitting window', 'type':'str', 'value':'0,0'})
#         opts['children'] = c
#         super(self.__class__, self).__init__(**opts)
#         #self.child('Fit').sigActivated.connect(self.fit_model)
#
#
#     def update(self, parameter):
#         self.fit_model()
#         CustomParameter.update(self, parameter)
#
#
#     def fit_model(self):
#         try:
#             ref_name = self.child('Reference data').value()
#             if ref_name is 'None':
#                 raise Exception('No reference data selected.')
#
#             names = list([s.name for s in experiment.scans])
#             scan = experiment.scans[names.index(ref_name)]
#             elastic = experiment.elastic_scans[names.index(ref_name)]
#             self.model.set_data(scan, elastic)
#
#             offset = self.child('Vertical offset').value()
#             self.model.set_voffset(offset)
#             #
#             # window_str = self.child('Fitting window').value()
#             # pattern = r'([-+]?\d*\.\d+|[-+]?\d+)'
#             # numbers = re.findall(pattern, window_str)
#             # window = list([float(n) for n in numbers])
#
#             wstart = float(self.child('Window (start)').value())
#             wend = float(self.child('Window (end)').value())
#
#             self.model.set_window([wstart, wend])
#
#             self.model.fit(experiment.analyzers, method = 'pearson7')
#
#         except Exception as e:
#             Log.error('Model fitting failed: {}'.format(e))
#
#
#     def update_lists(self):
#         d = dict()
#         d['Reference data'] = list([s.name for s in experiment.scans])
#         super(self.__class__, self).update_lists(d)
#
# registerParameterType('bgModel', BGModelParameter, override=True)


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

            for child in self.children():
                if child.name() == par.name():
                    child.remove()

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
        include = True, e_offset = 0.0, calibration = 'None'):
        opts = {}
        opts['name'] = 'Analyzer {}'.format( len(experiment.analyzers) + 1)
        opts['gui'] = self.opts['gui']
        opts['position'] = position
        opts['size'] = size
        opts['angle'] = angle
        opts['include'] = include
        opts['calibration'] = calibration
        opts['offset'] = e_offset
        super(self.__class__, self).addNew(**opts)
        # name=name, type=self.opts['child_type'], gui=self.opts['gui']

registerParameterType('analyzerGroup', AnalyzerGroupParameter, override=True)


class ScanGroupParameter(CustomGroupParameter):
    def __init__(self, **opts):
        opts['addText'] = 'Add scan'
        super(self.__class__, self).__init__(**opts)


    def addNew(self, scan = None, include = True, scanning_type = False,
        monitor_sum = True, offset_x = 0, offset_y = 0):
        """
        Will switch to interactive mode and ask for a scan to open if no scan is
        provided (i.e. scan = None).
        """


        if scan is None:
            scan = self.opts['gui'].action_read_scan()

        if scan is None:
            return

        opts = {}
        opts['scan'] = scan
        opts['name'] = scan.name
        opts['include'] = include
        opts['scanning_type'] = scanning_type
        opts['monitor_sum'] = monitor_sum
        opts['offset_x'] = offset_x
        opts['offset_y'] = offset_y

        # opts['bg_model_name'] = bg_model

        super(self.__class__, self).addNew(**opts)


registerParameterType('scanGroup', ScanGroupParameter, override=True)


class BGRoiGroupParameter(CustomGroupParameter):
    def __init__(self, **opts):
        opts['addText'] = 'Add background ROI'
        super(self.__class__, self).__init__(**opts)

    def addNew(self, position = [128,128], size = [20,20],
        include = True):
        opts = {}
        opts['name'] = 'Background ROI {}'.format(len(experiment.bg_rois) + 1)
        opts['gui'] = self.opts['gui']
        opts['position'] = position
        opts['size'] = size
        opts['include'] = include
        super(self.__class__, self).addNew(**opts)


registerParameterType('backgroundRoiGroup', BGRoiGroupParameter, override=True)


class CalibrationGroupParameter(CustomGroupParameter):
    def __init__(self, **opts):
        opts['addText'] = 'Add energy calibration'
        super(self.__class__, self).__init__(**opts)

    def addNew(self, scan_name = 'None', main_analyzer = None,
        first_frame = 0, last_frame = 0):
        opts = {}
        opts['name'] = 'Calibration {}'.format(len(experiment.calibrations) + 1)
        opts['elastic_scan'] = scan_name
        opts['first_frame'] = first_frame
        opts['last_frame'] = last_frame
        opts['main_analyzer'] = main_analyzer
        super(self.__class__, self).addNew(**opts)


registerParameterType('calibrationGroup', CalibrationGroupParameter, override=True)
