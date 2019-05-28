import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import matplotlib.cm as cm

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
