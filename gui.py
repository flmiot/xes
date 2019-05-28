import os
import sys

# ========================================================================
#   This is only needed for development.
#   Update the UI and resource file in case there were changes.
try:
    if sys.argv[1] == 'dev':
        os.system("pyrcc5 development/resources.qrc -o resources_rc.py")
        os.system("pyuic5 development/designer.ui -o xes/qt/ui.py")
        print("Done: Rebuilding of UI and resource file")
except:
    print("No rebuilding of UI and resource file.")
# ========================================================================

from pyqtgraph.Qt import QtCore

import xes
if __name__ == '__main__':

    if len(sys.argv) > 1 and sys.argv[1] != 'dev':
        # Schedule processing of input file
        timer = QtCore.QTimer()
        timer.singleShot(2000, lambda : w.parse_input_file(sys.argv[1]))

    xes.app.exec_()
