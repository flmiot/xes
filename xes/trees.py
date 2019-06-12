from pyqtgraph.parametertree import Parameter, ParameterTree, parameterTypes

import xes
from xes.parameters import ScanParameter


class ScanTree(ParameterTree):

    def __init__(self, parent):
        super(ScanTree, self).__init__(showHeader = False)
        par = Parameter.create(type='scanGroup', child_type='scan')
        self.setParameters(par, showTop=False)
        par.sigUpdate.connect(self.handler)


    def handler(self, parameter):
        monitor = xes.gui.monitor1
        plot    = xes.gui.plot

        if isinstance(parameter, ScanParameter):
            summed = parameter.child('Monitor: SUM').value()
            if parameter.scan.loaded:
                monitor.display(parameter.scan, sum = summed)

        if isinstance(parameter, parameterTypes.SimpleParameter):
            if parameter.name() == 'Monitor: SUM':
                summed = parameter.value()
                monitor.display(parameter.parent().scan, sum = summed)

            if parameter.name() == 'Include':
                plot.update_plot()

    def hide_all_scans(self):
        pass

    def invert_scan_selection(self):
        pass

    def solo_selected_scan(self):
        pass




class RoiTree(ParameterTree):

    def __init__(self, parent):
        super(RoiTree, self).__init__(showHeader = False)
        par = Parameter.create(type='analyzerGroup', child_type='analyzer')
        self.setParameters(par, showTop=False)
        par.sigUpdate.connect(self.handler)


    def handler(self, parameter):
        pass

    def hide_all_analyzers(self):
        pass

    def invert_analyzer_selection(self):
        pass

    def solo_selected_analyzer(self):
        pass



class BGRoiTree(ParameterTree):

    def __init__(self, parent):
        super(BGRoiTree, self).__init__(showHeader = False)
        par = Parameter.create(type='backgroundRoiGroup', child_type='backgroundRoi')
        self.setParameters(par, showTop=False)
        par.sigUpdate.connect(self.handler)


    def handler(self, parameter):
        pass

    def hide_all_bgrois(self):
        pass

    def polyfit_no_bgroi(self):
        pass

    def polyfit_all_bgrois(self):
        pass


class CalibrationTree(ParameterTree):

    def __init__(self, parent):
        super(CalibrationTree, self).__init__(showHeader = False)
        par = Parameter.create(type='calibrationGroup', child_type='manualCalibration')
        self.setParameters(par, showTop=False)
        par.sigUpdate.connect(self.handler)


    def handler(self, parameter):
        param = xes.gui.tree_runs.invisibleRootItem().child(0).param
        param.update_lists()
