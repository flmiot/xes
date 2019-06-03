import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

import xes
from xes.analysis import Analyzer, CalibrationRoi

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


class ManualCalibrationROI(pg.ROI):
    def __init__(self, *args, **kwargs):
        print(*args, **kwargs)
        # Use a custom pen to differentiate this ROI from the signal ROIs
        c = QtGui.QColor(*[0, 255, 0])
        pen = pg.mkPen(color=c)
        pg.ROI.__init__(self, *args, **kwargs)
        self.addScaleHandle([1, 1], [0, 0])
        print(*args, **kwargs)
        self.roi = CalibrationRoi(self, *args, **kwargs)
        x0,y0 = position[0] - size[0], position[1] - size[1]
        x1,y1 = position[0] + size[0], position[1] + size[1]
        self.roi.set_roi([x0,y0,x1,y1])
        self.roi.set_mask( mask = [195, 487] )
        # xes.experiment.add_background_roi(self.analyzer)
        # self.setToolTip(self.calibrationRoi.name)
        monitor.add_calibration_roi(self)
