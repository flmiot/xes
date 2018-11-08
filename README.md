# X-ray Raman data analysis tool (Win / Linux / macOS)

![alt text](https://github.com/flmiot/ixs/blob/master/doc/screenshot.PNG)

## Prerequisites
-	Python 3.4 or higher 
-	Non-standard python package PyQtGraph (http://www.pyqtgraph.org/)

It is recommended to use Anaconda (https://www.anaconda.com/) and run 
```
conda install -c anaconda pyqtgraph
```

To enter the GUI, just type
```
python gui.py [path/to/input_file.txt]
```

## Input files
Recognized plotting keywords are:
- !PLOT
- !NORMALIZE
- !SINGLE_SCANS
- !SINGLE_ANALYZERS
- !SUBTRACT_MODEL

Recognized analysis keywords are
- !SCANS
- !ANALYZERS
- !MODELS

Specifiy 
**scans** as:
```
scan(path, include, elastic-scan, monitor-sum, model)
```
**analyzers** as: 

```
analyzer(include, position-x, position-y, width, height, angle, pixel-wise, energy-offset)
```

**models** as:

```
model(reference-data, window-start, window-end,	vertical-offset)
``` 

This is shown in the example below:

```
! SCANS
scan(
	path='H:\raw\alignment_00887',
	include=False,
	elastic-scan='alignment_00887',
	monitor-sum=True,
	model='None'
	)
scan(
	path='H:\raw\alignment_00888',
	include=True,
	elastic-scan='alignment_00887',
	monitor-sum=True,
	model='Model 1'
	)

! ANALYZERS
analyzer(
	include=True,
	position-x = 80,
	position-y= 80,
	width = 100,
	height = 50,
	angle = 45,
	pixel-wise=False,
	energy-offset=45.9
	)
	

! MODELS
model(
	reference-data='alignment_00888',
	window-start=95,
	window-end=450,
	vertical-offset=0
	)
	
! PLOT
! NORMALIZE
```
