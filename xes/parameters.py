import re
import logging

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, parameterTypes, registerParameterType

import xes
from xes.analysis import Analyzer, Calibration

Log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def register_parameter_types():
    registerParameterType('calibrationGroup', CalibrationGroupParameter, override=True)
    registerParameterType('backgroundRoiGroup', BGRoiGroupParameter, override=True)
    registerParameterType('scanGroup', ScanGroupParameter, override=True)
    registerParameterType('analyzerGroup', AnalyzerGroupParameter, override=True)
    registerParameterType('calibration', CalibrationParameter, override=True)
    registerParameterType('scan', ScanParameter, override=True)
    registerParameterType('backgroundRoi', BackgroundParameter, override=True)
    registerParameterType('analyzer', AnalyzerParameter, override=True)


class CustomParameterItem(parameterTypes.GroupParameterItem):
    def __init__(self, *args, **kwargs):
        self.toplevel_color = QtGui.QColor(214, 239, 255)
        super(self.__class__, self).__init__(*args, **kwargs)
        self.contextMenu.addAction("Get snippet").triggered.connect(self.snippet)

    def snippet(self):
        self.param.snippet()

    def contextMenuEvent(self, ev):
        self.contextMenu.popup(ev.globalPos())


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
    sigSnippet      = QtCore.Signal(object)


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


    def snippet(self, snippet_dict = None):
        """Called from inherited class"""
        if snippet_dict is not None:
            content = ''
            for property in snippet_dict['properties']:
                value = snippet_dict['properties'][property]
                content +='\n\t{}={}'.format(property, value)

            fmt = '{}({}\n)'.format(snippet_dict['property_name'], content)
            self.sigSnippet.emit(fmt)


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
        x0,y0 = position[0] - size[0], position[1] - size[1]
        x1,y1 = position[0] + size[0], position[1] + size[1]
        self.analyzer.set_roi([x0,y0,x1,y1])
        self.analyzer.set_mask( mask = [195, 487] )
        xes.experiment.add_analyzer(self.analyzer)
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
            xes.gui.monitor1) # opts['angle'],

        c = []
        c.append({'name': 'Include', 'type':'bool', 'value':opts['include']})
        # names = list([c.name for c in experiment.calibrations])
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


    def snippet(self):
        snippet_dict = {
            'property_name' : 'analyzer',
            'properties'    : {
            	'include'      : self.child('Include').value(),
            	'position-x'   : self.roi.pos()[0],
            	'position-y'   : self.roi.pos()[1],
            	'width'        : self.roi.size()[1],
            	'height'       : self.roi.size()[0]
                }
        	}
        super(self.__class__, self).snippet(snippet_dict)




class BackgroundROI(pg.ROI):
    def __init__(self, name, position, size, monitor): # angle,
        # Make this a different color

        # Use a custom pen to differentiate this ROI from the signal ROIs
        c = QtGui.QColor(*[177, 206, 198])
        pen = pg.mkPen(color=c, style=QtCore.Qt.DashLine)

        pg.ROI.__init__(self, pos=position, size=size, pen = pen) #, scaleSnap=True, translateSnap=True)
        self.addScaleHandle([1, 1], [0, 0])
        self.addScaleHandle([0, 0], [1, 1])
        self.addScaleHandle([0, 1], [1, 0])
        self.addScaleHandle([1, 0], [0, 1])

        self.addScaleHandle([0.5, 1], [0.5, 0])
        self.addScaleHandle([0.5, 0], [0.5, 1])

        self.analyzer = Analyzer(name)
        x0,y0 = position[0] - size[0], position[1] - size[1]
        x1,y1 = position[0] + size[0], position[1] + size[1]
        self.analyzer.set_roi([x0,y0,x1,y1])
        self.analyzer.set_mask( mask = [195, 487] )
        xes.experiment.add_background_roi(self.analyzer)
        self.setToolTip(self.analyzer.name)
        monitor.add_background_roi(self)


class BackgroundParameter(CustomParameter):
    def __init__(self, **opts):
        self.roi = BackgroundROI(opts['name'], opts['position'], opts['size'],
            xes.gui.monitor1) # opts['angle'],

        c = []
        c.append({'name': 'Include', 'type':'bool', 'value':opts['include']})
        c.append({'name': 'Polynomial fit', 'type':'bool', 'value':opts['polyfit']})
        c.append({'name': 'Polynomial order', 'type':'int', 'value':opts['polyorder']})
        opts['children'] = c
        super(self.__class__, self).__init__(**opts)


    def update(self, parameter):
        self.roi.analyzer.active = self.child('Include').value()
        self.roi.analyzer.poly_fit = self.child('Polynomial fit').value()
        self.roi.analyzer.poly_order = self.child('Polynomial order').value()
        super(self.__class__, self).update(parameter)


    def snippet(self):
        snippet_dict = {
            'property_name' : 'background',
            'properties'    : {
            	'include'      : self.child('Include').value(),
            	'position-x'   : self.roi.pos()[0],
            	'position-y'   : self.roi.pos()[1],
            	'width'        : self.roi.size()[1],
            	'height'       : self.roi.size()[0]
                }
        	}
        super(self.__class__, self).snippet(snippet_dict)



class ScanParameter(CustomParameter):
    """ Scan parameter class represents a scan object. Parameters are

    ------------------  --------------------------------------------------
    Include             Include this object in analysis
    Scanning type       Treat as scan (e.g. HERFD)
    Monitor: SUM        When displaying this object in monitor, sum all images
    ------------------  --------------------------------------------------
    """

    def __init__(self, **opts):
        scan = opts['scan']

        include = opts['include']
        scanning_type = opts['scanning_type']
        monitor_sum = opts['monitor_sum']
        offset_x = opts['offset_x']
        offset_y = opts['offset_y']
        elastic = opts['elastic']
        elastic_range = opts['elastic_range']


        xes.experiment.add_scan(scan)
        names = ['None']
        names.extend(list([s.name for s in xes.experiment.scans]))

        if not elastic in names:
            names.append(elastic)

        c = []
        c.append({'name': 'Include', 'type':'bool', 'value':include})
        c.append({'name': 'Monitor: SUM', 'type':'bool', 'value':monitor_sum})
        c.append({'name': 'Elastic scan', 'type':'list', 'values': names,
            'value':elastic})
        c.append({'name': 'Range', 'type':'str', 'value': elastic_range})
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

        calibration = Calibration()
        elastic_name = self.child('Elastic scan').value()

        # print(self.child('Elastic range').value())
        matches = re.findall(r'(\d+)', self.child('Range').value())
        if self.scan.loaded:
            self.scan.range = list([int(d) for d in matches])
            self.scan.range[1] += 1

        if elastic_name != "None":
            ind = list([s.name for s in xes.experiment.scans]).index(elastic_name)
            elastic_scan = xes.experiment.scans[ind]
            calibration.elastic_scan = elastic_scan

        xes.experiment.change_calibration(self.scan, calibration)

            # ind = list([s.name for s in experiment.scans]).index(self.scan.name)
            # experiment.elastic_scans[ind] = elastic_scan


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


    def update_lists(self):
        d = dict()
        elastic_scans = ['None']
        elastic_scans.extend(list([s.name for s in xes.experiment.scans]))
        d['Elastic scan'] = elastic_scans
        # l = ['None']
        # l.extend(list([b.name for b in experiment.bg_models]))
        # d['Background model'] = l
        super(self.__class__, self).update_lists(d)


    def snippet(self):
        snippet_dict = {
            'property_name' : 'scan',
            'properties'    : {
            	'path'         : self.scan.log_file,
            	'include'      : self.child('Include').value(),
            	'elastic-scan' : self.child('Elastic scan').value(),
            	'monitor-sum'  : self.child('Monitor: SUM').value(),
            	'range'        : self.child('Range').value()
                }
        	}
        super(self.__class__, self).snippet(snippet_dict)



class CalibrationParameter(CustomParameter):
    def __init__(self, **opts):
        self.calibration = Calibration()
        self.calibration.name = opts['name']
        xes.experiment.add_calibration(self.calibration)
        names = ['None']
        names.extend(list([scan.name for scan in xes.experiment.scans]))
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



class CustomGroupParameter(parameterTypes.GroupParameter):

    sigUpdate = QtCore.Signal(object)
    sigItemSelected = QtCore.Signal(object)
    sigSnippet      = QtCore.Signal(object)

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
            child.sigSnippet.connect(self.snippet)


    def update(self, parameter):
        self.sigUpdate.emit(parameter)


    def snippet(self, object):
        self.sigSnippet.emit(object)


class AnalyzerGroupParameter(CustomGroupParameter):
    def __init__(self, **opts):
        opts['addText'] = 'Add analyzer'
        super(self.__class__, self).__init__(**opts)


    def addNew(self, position = [128,128], size = [20,20], angle = 0.0,
        include = True, e_offset = 0.0, calibration = 'None'):
        opts = {}
        opts['name'] = 'Analyzer {}'.format( len(xes.experiment.analyzers) + 1)
        opts['position'] = position
        opts['size'] = size
        opts['angle'] = angle
        opts['include'] = include
        opts['calibration'] = calibration
        opts['offset'] = e_offset
        super(self.__class__, self).addNew(**opts)
        # name=name, type=self.opts['child_type'], gui=self.opts['gui']


class ScanGroupParameter(CustomGroupParameter):
    def __init__(self, **opts):
        opts['addText'] = 'Add run'
        super(self.__class__, self).__init__(**opts)


    def addNew(self, scan = None, include = True, scanning_type = False,
        monitor_sum = True, elastic = "None", offset_x = 0, offset_y = 0,
        range = None, slices = 5):
        """
        Will switch to interactive mode and ask for a scan to open if no scan is
        provided (i.e. scan = None).
        """


        if scan is None:
            scan = xes.gui.add_scan()

        if scan is None:
            return

        if range is None:
            range = '{},{}'.format(0, len(scan.files)-1)

        opts = {}
        opts['scan'] = scan
        opts['name'] = scan.name
        opts['include'] = include
        opts['scanning_type'] = scanning_type
        opts['monitor_sum'] = monitor_sum
        opts['offset_x'] = offset_x
        opts['offset_y'] = offset_y
        opts['elastic'] = elastic
        opts['elastic_range'] = range

        super(self.__class__, self).addNew(**opts)
        self.update_lists()



class BGRoiGroupParameter(CustomGroupParameter):
    def __init__(self, **opts):
        opts['addText'] = 'Add background ROI'
        super(self.__class__, self).__init__(**opts)

    def addNew(self, position = [128,128], size = [20,20],
        include = True, polyfit = False, polyorder = 6):
        opts = {}
        opts['name'] = 'Background ROI {}'.format(len(xes.experiment.bg_rois) + 1)
        opts['position'] = position
        opts['size'] = size
        opts['include'] = include
        opts['polyfit'] = polyfit
        opts['polyorder'] = polyorder
        super(self.__class__, self).addNew(**opts)




class CalibrationGroupParameter(CustomGroupParameter):
    def __init__(self, **opts):
        opts['addText'] = 'Add energy calibration'
        super(self.__class__, self).__init__(**opts)

    def addNew(self, scan_name = 'None', main_analyzer = None,
        first_frame = 0, last_frame = 0):
        opts = {}
        opts['name'] = 'Calibration {}'.format(len(xes.experiment.calibrations) + 1)
        opts['elastic_scan'] = scan_name
        opts['first_frame'] = first_frame
        opts['last_frame'] = last_frame
        opts['main_analyzer'] = main_analyzer
        super(self.__class__, self).addNew(**opts)
