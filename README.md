# X-ray emission & HERFD data analysis tool (Win / Linux / macOS)

![alt text](https://github.com/flmiot/xes/blob/master/doc/screenshot.PNG)

![alt text2](https://github.com/flmiot/xes/blob/master/doc/screenshot2.PNG)

## Prerequisites
-	Python 3.4 or higher
-	Non-standard python package __PyQtGraph__ (http://www.pyqtgraph.org/), written by Luke Campagnola

It is recommended to use Anaconda (https://www.anaconda.com/) and run
```
conda install -c anaconda pyqtgraph
```

To enter the GUI, just type
```
python gui.py 
```

## Features:
- Manage numerous scans and analyzer (ROIs) at the same time. View them separately, overlayed or summed (via linear interpolation) 
- Apply background subtraction by setting background ROIs. Appropriate background ROIs are automatically selected for each analyzer ROI (closest above and below)
- Specify an elastic scan to calibrate the energy axis for each analyzer
- Enable the automatic plotting-update to view and access changes quickly (e.g ROI size) 
- Switch to HERFD mode: Display emission data along the incoming energy axis and visualize XAS data which was taken in flourescence mode
