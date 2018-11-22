import os
import re
# import imageio
import tifffile as tiff
import pdb
import time
import numpy as np
import logging

import scipy.optimize as optim
import scipy.interpolate as interp
from scipy.signal import correlate

import matplotlib.pyplot as plt # For debugging

Log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class Experiment(object):
    """An Experiment holds a set of analyzers for a number of scans. For each
    scan, summed spectra can be obtained by calling the *get_spectrum()* method.
    The returned spectra can be summed over all active analyzers (property
    'active' of each analyzer). Additionally, all scans can be summed for which
    the 'active' property was set to true (see *Scan* class).
    """

    def __init__(self):
        """Use the *add_scan* method to populate the experiment with scans."""
        self.scans                  = [] # List of all added scans
        self.analyzers              = [] # List of all added anaylzers
        self.bg_rois                = [] # List of all available background ROIs
        self.calibrations         = [] # List of energy calibrations



    def get_spectrum(self, single_analyzers = True, single_scans = True):

        """
        Get an energy spectrum. This method will return three arrays, one with
        energy axes, one with corresponding spectra and one with backgrounds. If
        *single_analyzers* is set to False, sum over all analyzers with property
        'active' set to True (see *Analyzer* class). If *single_scans* is set to
        False, sum over all scans with property 'active' set to True (see *Scan*
        class). Returned arrays will contain just one energy axis and one
        spectrum if both *single_analyzers* and *single_scans* are set to False.
        """


        start = time.time()
        active_analyzers = list([a for a in self.analyzers if a.active])
        active_scans = list([s for s in self.scans if s.active])

        if len(active_scans) < 1:
            raise ValueError("No active scans!")
        if len(active_analyzers) < 1:
            raise ValueError("No active analyzers!")

        types = list(
            [(s.name,'{}f4'.format(len(s.images))) for s in active_scans])

        e = np.empty(len(active_analyzers), types)
        i = np.empty(len(active_analyzers), types)
        b = np.empty(len(active_analyzers), types)

        for scan in active_scans:

            index = self.scans.index(scan)

            # Center analyzers with the corresponding elastic scan
            elastic_scan = self.elastic_scans[index]
            elastic_scan.center_analyzers(active_analyzers)

            ener, inte, back = scan.get_energy_loss_spectrum(active_analyzers)
            e[scan.name] = ener
            i[scan.name] = inte
            b[scan.name] = back

            # plt.plot(back[0])
            # plt.show()

        end = time.time()
        fmt = "Single spectra obtained [Took {:2f} s]".format(end-start)
        Log.debug(fmt)
        start = end

        if not single_analyzers:
            try:
                e_summed, i_summed, b_summed = self.sum_analyzers(e, i, b)
                e, i, b = e_summed, i_summed, b_summed
            except Exception as exception:
                Log.error('Summing of analyzers failed: {}'.format(exception))

        if not single_scans:
            try:
                e_summed, i_summed, b_summed = self.sum_scans(e, i, b)
                e, i, b = e_summed, i_summed, b_summed
            except Exception as exception:
                Log.error('Summing of scans failed: {}'.format(exception))


        end = time.time()
        fmt = "Spectra summed [Took {:2f} s]".format(end-start)
        Log.debug(fmt)

        return e, i, b


    def add_analyzer(self, analyzer):
        """Add an analyzer object to this experiment. Raise an exception if
        analyzer already exists.
        """

        if analyzer in self.analyzers:
            raise ValueError("Analyzer is already used for this experiment.")

        self.analyzers.append(analyzer)


    def remove_analyzer(self, analyzer):
        """Remove an analyzer from the experiment."""
        if analyzer not in self.analyzers:
            raise ValueError("Analyzer is not known inside this experiment.")

        self.analyzers.remove(analyzer)

    def add_calibration(self, calibration):
        self.calibrations.append(calibration)

    def add_background_roi(self, bg_roi):
        """Add an bg roi object to this experiment. Raise an exception if
        bg roi already exists.
        """

        if bg_roi in self.bg_rois:
            raise ValueError("Background ROI is already used for this experiment.")

        self.bg_rois.append(bg_roi)


    def add_scan(self, scan):
        """Add a scan object. Specify scan and corresponding elastic scan. Raise
        and exception if a scan with same name already exists inside this
        experiment.
        """

        for s in self.scans:
            if s.name == scan.name:
                raise ValueError('Scan could not be added to the experiment '\
                    'because the name already exists.')

        self.scans.append(scan)


    def remove_scan(self, scan_name):
        """Add a scan object with name *scan_name*. Corresponding elastic scan
        is removed from this experiment automatically.
        """

        index = None
        for s in self.scans:
            if s.name == scan_name:
                index = self.scans[s]

        if index is not None:
            self.elastic_scans.remove(self.elastic_scans[index])
            self.scans.remove(self.scans[index])
            self.backgrounds.remove(self.backgrounds[index])
        else:
            raise ValueError("Unknown scan requested for removal.")


    # def add_bg_model(self, bg_model):
    #     self.bg_models.append(bg_model)


    def change_elastic_scan(self, scan, elastic_scan_name):
        """Replace current elastic scan for *scan* with
        *elastic_scan_name*. Raise an exception, if *scan* is unknown.
        """
        if scan not in self.scans:
            raise ValueError("Elastic scan could not be set for scan. Scan "\
                "unknown.")

        names = list([s.name for s in self.scans])
        if elastic_scan_name in names:
            elastic_scan = self.scans[names.index(elastic_scan_name)]
        else:
            raise ValueError("Unknown elastic scan requested.")

        self.elastic_scans[self.scans.index(scan)] = elastic_scan


    def sum_analyzers(self, energies, intensities):
        """
        Interpolate spectra for multiple analyzers linearly and sum them
        scan-wise.
        """

        scan_names = energies.dtype.names

        # Allocate memory
        types = []
        for scan_name in scan_names:
            points = np.max([len(e) for e in energies[scan_name]])
            types.append((scan_name, '{}f4'.format(points)))

        ce = np.empty(1, dtype = types)
        ii = np.zeros(1, dtype = types)

        for scan_name in scan_names:

            scan_energies = energies[scan_name]
            scan_intensities = intensities[scan_name]

            min_energy = np.max(list(e[0] for e in scan_energies))
            max_energy = np.min(list(e[-1] for e in scan_energies))
            points = np.max(list([len(e) for e in scan_energies]))

            ce[scan_name] = np.linspace(min_energy, max_energy, points)

            a = len(energies)
            z = zip(range(a), scan_energies, scan_intensities)
            for ind,e,i in z:
                fi = interp.interp1d(e, i)
                ii[scan_name] += fi(ce[scan_name])

        return ce, ii


    def sum_scans(self, energies, intensities):
        """
        Interpolate spectra for multiple scans linearly and sum them
        analyzer-wise.
        """

        scan_names = energies.dtype.names

        # Allocate memory
        points = np.max([len(energies[name][0]) for name in scan_names])
        types = [('Summed scans', '{}f4'.format(points))]

        ce = np.empty(len(energies), dtype=types)
        ii = np.zeros(len(energies), dtype=types)
        bi = np.zeros(len(energies), dtype=types)

        a = len(energies)
        z = zip(range(a), energies, intensities, backgrounds)
        for ind, energy, intensity, background in z:
            min_energy = np.max([energy[name][0] for name in scan_names])
            max_energy = np.min([energy[name][-1] for name in scan_names])

            ce['Summed scans'][ind] = \
                np.linspace(min_energy, max_energy, points)

            for name in scan_names:
                e = energy[name]
                i = intensity[name]
                fi = interp.interp1d(e, i)
                ii['Summed scans'][ind] += fi(ce['Summed scans'][ind])

                if backgrounds is not None:
                    b = background[name]
                    fb = interp.interp1d(e,b)
                    bi['Summed scans'][ind] += fb(ce['Summed scans'][ind])

        return ce, ii, bi


class Scan(object):
    def __init__(self, log_file, imgage_files):
        """ Specify *logfile* as namestring and list of *image_files*.
        Assumes that logfile holds energy for each image.
        """
        self.name       = os.path.splitext(os.path.split(log_file)[1])[0]
        self.log_file   = log_file
        self.files      = imgage_files
        self.images     = None
        self.energies   = None
        self.monitor    = None
        self.active     = True
        # self.bg_model   = None
        self.loaded     = False
        self.offset     = [0,0]


    @property
    def images(self):
        return self.__images


    @images.setter
    def images(self, images):
        self.__images = images


    def read_logfile(self):
        with open(self.log_file, 'r') as content_file:
            content = content_file.read()

        print(content)
        # pattern = r'(\d+\.\d+)\s'*8+r'.*'+r'(\d+\.\d+)\s'*3
        pattern = r'([+-]*\d+\.\d+[e0-9-]*)\s'*14
        matches = re.findall(pattern, content)
        enc_dcm_ener = [0]*len(matches)
        pin2 = [0]*len(matches)
        for ind, match in enumerate(matches):
            _,_,_,e,i01,tfy,trans,i02,_,_,_,_,_,_ = match
            enc_dcm_ener[ind] = float(e)
            i01 = float(i01)
            i02 = float(i02)
            tfy = float(tfy)
            trans = float(trans)

        self.energies = enc_dcm_ener
        self.monitor = i01

        # plt.plot(pin2)
        # plt.show()


    def read_files(self, callback = None):

        self.images = np.empty(len(self.energies), dtype = '(195,487)i4')
        for ind, filename in enumerate(self.files):
            self.images[ind] = tiff.imread(filename)

            if callback is not None:
                callback(ind + 1)

        self.loaded = True

        # arr = np.array(self.images)
        # arr = np.sum(arr, axis = 0)
        # plt.imshow(np.log(arr))
        # plt.show()

    def save_scan(self, path, analyzers):
        """Save energy loss spectra of this scan object into a textfile."""
        arr = np.array(self.get_energy_loss_spectrum(analyzers))
        arr = np.reshape(arr, (2*len(analyzers),-1))

        header = ''
        for ind in range(len(analyzers)):
            header += 'ene_{0} ana_{0} '.format(ind)
        np.savetxt(path, arr.T, header = header, comments = '')


    def center_analyzers(self, analyzers):
        for an in analyzers:
            pixel_wise = an.pixel_wise
            an.determine_central_energy(images=self.images, base=self.energies,
                pixel_wise = pixel_wise)


    def get_energy_loss_spectrum(self, analyzers, bg_model = None):

        t = '{}f4'.format(len(self.images))
        energies = np.empty(len(analyzers), dtype = t)
        intensities = np.empty(len(analyzers), dtype = t)
        backgrounds = np.empty(len(analyzers), dtype = t)

        for ind, an in enumerate(analyzers):
            b, s = an.get_signal_series(images=self.images, base=self.energies)

            s /= self.monitor

            if bg_model is None:
                bg_model = self.bg_model

            if bg_model is not None:
                try:
                    bg = bg_model.get_background(b, an)
                except Exception as e:
                    Log.error('Background was not determined: {}'.format(e))
                    bg = np.zeros(len(s))
            else:
                bg = np.zeros(len(s))

            energies[ind] = b
            intensities[ind] = s
            backgrounds[ind] = bg

        return energies, intensities, backgrounds


    def set_background_model(self, bg_model):
        self.bg_model = bg_model


class Analyzer(object):
    def __init__(self, name):
        """
        Analyzers have a
        property called *active*, which can be used to include/exclude them when
        doing summed signal analysis.
        """

        self.calibration        = None
        self.energy_offset      = 0.0
        self.roi                = None      # xmin,ymin,xmax,ymax e.g. [0,0,5,5]
        self.active             = True
        self.name               = name


    def set_roi(self, roi):
        """Specify *pixels* as list of tuples"""

        if isinstance(roi, list):
            self.roi = np.array(list)
        elif isinstance(roi, np.ndarray):
            self.roi = roi
        else:
            fmt = "ROI has to be specified like [xmin, ymin, xmax, ymax], "\
                "either as list or np.ndarray."
            raise Exception(fmt)


    def set_calibration(self, calibration):
        self.calibration = calibration.register(self)


    def get_signal(self, image):
        """
        """

        if self.roi is None:
            raise ValueError("ROI needs to be set before use.")

        x0,y0,x1,y1 = self.roi
        ii = np.sum(image[y0:y1+1,x0:x1+1], axis = 0)

        try:
            ea = self.calibration.get_e_axis(self, signal, self.roi)
        except:
            ea = np.arange(len(signal))
            fmt = "Energy axis could not be determined for analyzer {}."
            Log.error(fmt.format(self.name))

        return ea, ii


    def get_signal_series(self, images):
        """

        """

        start = time.time()

        ea = np.empty((len(images), len(np.arange(x0, x1+1))))
        ii = np.empty((len(images), len(np.arange(x0, x1+1))))
        for ind, image in enumerate(images):
            e_axis, intensity = self.get_signal(image)
            ea[ind] = e_axis
            ii[ind] = intensity

        end = time.time()
        fmt = "Returned signal series [Took {:2f} s]".format(end-start)
        Log.debug(fmt)

        return ea, ii


class Calibration(object):
    def __init__(self):
        self.name           = None
        self.main_analyzer  = None
        self.elastic_scan   = None
        self.first_frame    = None
        self.last_frame     = None
        self.analyzers      = []
        self.offsets        = []


    def register(self, analyzer):

        if analyzer in self.analyzers:
            raise Exception("Analyzer already registered for this calibration.")

        self.analyzers.append(analyzer)
        self.offsets.append(0.0)
        self.calibrate(analyzer)
        return self


    def set_main_analyzer(self, analyzer):
        self.main_analyzer = analyzer
        self.calibrate(analyzer)


    def get_e_axis(self, analyzer):
        a = 1


    def calibrate(self, analyzer):
        if analyzer is self.main_analyzer:
            a = 1
        else:
            a = 1

#
# class BGModel(object):
#     def __init__(self):
#         """No of sample rows, No of sample columns"""
#         self.name       = None
#         self.scan       = None
#         self.elastic    = None
#         self.fits       = []
#         self.window     = None
#         self.guess      = None
#         self.offset     = 0.0
#
#     def set_data(self, scan, elastic):
#         self.scan = scan
#         self.elastic = elastic
#
#     def set_voffset(self, offset):
#         self.offset = offset
#
#     def set_window(self, window):
#         self.window = window
#
#     def set_fitting_guess(self, guess):
#         self.guess = guess
#
#     def get_background(self, x, analyzer):
#         """
#         Get the fitted background function evaluated for all values in
#         array *e*. Will check if a fitting method was called before use.
#         """
#
#         try:
#             index = self.analyzers.index(analyzer)
#             return self.fits[index](x) - self.offset
#
#         except:
#             raise Exception('No fit available for this analyzer.')
#
#
#     def fit(self, analyzers, method = 'pearson7'):
#         """"""
#         if self.scan is None:
#             raise Exception('No reference data for this background model.')
#
#         self.analyzers = analyzers
#         self.fits.clear()
#
#         # Set central energies for fit data
#         self.elastic.center_analyzers(analyzers)
#         e,i,_ = self.scan.get_energy_loss_spectrum(analyzers)
#
#         if method == 'pearson7':
#             for x,y in zip(e,i):
#                 f = self.fit_pearson7(x,y, fit_window = self.window)
#                 self.fits.append(f)
#
#             fmt = 'Model fitting (method = pearson7) sucessful.'
#             Log.debug(fmt)
#         else:
#             raise Exception('Unknown fitting method requested.')
#
#
#     def fit_pearson7(self, x, y, fit_window = None):
#         """"""
#         if fit_window is None:
#             x0,x1 = 0,len(x)
#         else:
#             x0 = np.argmin(np.abs(x - fit_window[0]))
#             x1 = np.argmin(np.abs(x - fit_window[1]))
#
#         if self.guess is None:
#             index = np.where(y == np.max(y[x0:x1]))[0]
#             en_at_max = x[index]
#             self.guess = [np.max(y[x0:x1]), 60, 1.18, en_at_max]
#
#         def _pearson7(x, imax, w,m, x0):
#             return imax * w**(2*m) / (w**2+(2**(1/m)-1)*(x-x0)**2)**m
#
#         bv, _ = optim.curve_fit(_pearson7, x[x0:x1], y[x0:x1], p0=self.guess)
#
#         # plt.plot(_pearson7(x, *bv))
#         # plt.show()
#
#         return lambda e : _pearson7(e, *bv)
