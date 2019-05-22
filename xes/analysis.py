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



class Label(object):
    def __init__(self):
        self.labels = {}        # Dict of type {ScanName: [Analyzer1, ...], }

    def add_scan_labels(self, label_dict):
        self.labels.update(label_dict)

    def get_labels(self, single_scans = True, single_analyzers = True):
        """
        S           Number of scans
        A           Number of analyzers

        Return S x A array of labels if single_scans = False and
        single_analyzers = False

        Return S x 1 array of labels if single_scans = False and
        single_analyzers = True

        Return A x 1 array of labels if single_scans = True and
        single_analyzers = False

        Return single labels if single_scans = False and
        single_analyzers = False
        """

        lret = []
        if single_scans and single_analyzers:
            fmt = "{} - {}"
            for scan, analyzers in zip(self.labels, self.labels.values()):
                lret.append(list([fmt.format(scan, a) for a in analyzers]))

        elif single_scans and not single_analyzers:
            fmt = "{} - Summed: {}"
            for scan, analyzers in zip(self.labels, self.labels.values()):
                a_str = self.make_comma_separated_list(analyzers)
                lret.append([fmt.format(scan, a_str)])

        elif not single_scans and single_analyzers:
            scan_names = list(self.labels.keys())
            s_str = self.make_comma_separated_list(scan_names)
            first_key = list(self.labels.keys())[0]
            fmt = "{} - Summed: {}"
            lret.append(list([fmt.format(a, s_str) for a in self.labels[first_key]]))

        else:
            scan_names = list(self.labels.keys())
            s_str = self.make_comma_separated_list(scan_names)
            first_key = list(self.labels.keys())[0]
            analyzer_names = list([a for a in self.labels[first_key]])
            a_str = self.make_comma_separated_list(analyzer_names)
            lret.append(["Summed: {} - Summed: {}".format(s_str, a_str)])

        return np.array(lret)


    def make_comma_separated_list(self, labels):
        li = ""
        for ind, l in enumerate(labels):
            if ind == len(labels) - 1:
                li += l
            elif ind == 0 and ind == len(labels) - 1:
                li += l
            else:
                li += l + ", "

        return li


class AnalysisResult(object):
    def __init__(self):
        self.in_e           = []
        self.out_e          = []
        self.intensities    = []
        self.background     = []
        self.fits           = []
        self.labels         = Label()


    def add_data(self, in_e, out_e, intensity, background, fit, label_dict):
        self.in_e.append(in_e)
        self.out_e.append(out_e)
        self.intensities.append(intensity)
        self.background.append(background)
        self.fits.append(fit)
        self.labels.add_scan_labels(label_dict)


    def get_curves(
        self, single_scans, single_analyzers,
        scanning_type = False,
        single_image = None,
        slices = 1,
        normalize_scans_before_sum = False,
        normalize_analyzers_before_sum = False
        ):
        """
        S           Number of scans
        A           Number of analyzers
        P_IN        Number of points along e_in axis
        """

        in_e, out_e = self.in_e, self.out_e
        i, b = self.intensities, self.background
        l = self.labels.get_labels(single_scans, single_analyzers)

        no_scans = len(i)
        no_analyzers = len(i[0])
        no_points_in_e = len(i[0][0])

        if scanning_type:
            ii = np.empty((no_scans, no_analyzers), dtype = list)
            bi = np.empty((no_scans, no_analyzers), dtype = list)
            ei = np.empty((no_scans, no_analyzers), dtype = list)

            # Iterate scans
            z = zip(range(len(i)), i, b)
            for ind_s, il, bl in z:
                # Iterate analyzers
                for ind_a in range(no_analyzers):
                    g1 = [np.sum(img) for img in il[ind_a]]
                    g2 = [np.sum(img) for img in bl[ind_a]]
                    ii[ind_s, ind_a] = np.array(g1)
                    bi[ind_s, ind_a] = np.array(g2)
                    ei[ind_s, ind_a] = np.array(in_e[ind_s])

        else:
            ii = []
            bi = []

            # Iterate scans
            z = zip(range(len(i)), i, b)
            for ind, il, bl in z:
                if not single_image is None:
                    if slices == 1:
                        ii.append(il[:, single_image])
                        bi.append(bl[:, single_image])
                    else:
                        i0 = single_image - int(slices / 2)
                        i1 = i0 + slices
                        if i0 < 0:
                            i0 = 0

                        Log.debug("Plotting slices from {} - {}".format(i0, i1))
                        ii.append(np.sum(il[:, i0:i1], axis = 1))
                        bi.append(np.sum(bl[:, i0:i1], axis = 1))
                else:
                    ii.append(np.sum(il, axis = 1))
                    bi.append(np.sum(bl, axis = 1))
            ii = np.array(ii)
            bi = np.array(bi)
            ei = np.array(out_e)

        if not single_analyzers:
            ei, ii, bi = self.sum_analyzers(ei, ii, bi, normalize_analyzers_before_sum)

        if not single_scans:
            ei, ii, bi = self.sum_scans(ei, ii, bi, normalize_scans_before_sum)

        return ei, ii, bi, l


    def sum_analyzers(self, energies, intensities, backgrounds, normalize_before_sum = False):
        """
        Interpolate spectra for multiple analyzers linearly and sum them
        scan-wise.

        S               Number of scans (of 1 if summed)
        A               Number of analyzers (of 1 if summed)
        P               Number of points along energy axis

        energies        Array (S x A x P)
        intensities     Array (S x A x P)
        backgrounds     Array (S x A x P)
        """

        energies_summed = np.empty((len(energies),1), dtype = list)
        intensities_summed = np.empty((len(energies),1), dtype = list)
        backgrounds_summed = np.empty((len(energies),1), dtype = list)


        # Iterate over scans
        z = zip(range(len(energies)), energies, intensities, backgrounds)
        for ind, energy, intensity, background in z:

            ce, ii, b = self._interpolate_and_sum(energy, intensity, background, normalize_before_sum)

            energies_summed[ind] = [ce]
            intensities_summed[ind] = [ii]
            backgrounds_summed[ind] = [b]

        return energies_summed, intensities_summed, backgrounds_summed


    def sum_scans(self, energies, intensities, backgrounds, normalize_before_sum = False):
        """
        Interpolate spectra for multiple scans linearly and sum them
        analyzer-wise.

        S               Number of scans (of 1 if summed)
        A               Number of analyzers (of 1 if summed)
        P               Number of points along energy axis

        energies        Array (S x A x P)
        intensities     Array (S x A x P)
        backgrounds     Array (S x A x P)
        """

        n_analyzers = len(energies[0])
        energies_summed = np.empty((1, n_analyzers), dtype = list)
        intensities_summed = np.empty((1, n_analyzers), dtype = list)
        backgrounds_summed = np.empty((1, n_analyzers), dtype = list)

        # Iterate over analyzers
        z = zip(range(n_analyzers), energies.T, intensities.T, backgrounds.T)
        for ind, energy, intensity, background in z:
            ce, ii, b = self._interpolate_and_sum(energy, intensity, background, normalize_before_sum)

            energies_summed.T[ind]      = [ce]
            intensities_summed.T[ind]   = [ii]
            backgrounds_summed.T[ind]   = [b]

        return energies_summed, intensities_summed, backgrounds_summed


    def _interpolate_and_sum(self, energy, intensity, background, normalize_before_sum = False):

        min_energy = np.max(list(np.min(e) for e in energy))
        max_energy = np.min(list(np.max(e) for e in energy))
        # print("min_energy", min_energy, "max_energy", max_energy)

        points = np.max(list([len(i) for i in intensity]))
        ii = np.zeros(points, dtype = np.float)
        bg = np.zeros(points, dtype = np.float)
        ce = np.linspace(min_energy, max_energy, points)

        for e, i, b in zip(energy, intensity, background):

            fi = interp.interp1d(e, i)
            fb = interp.interp1d(e, b)
            if normalize_before_sum:
                b       = fb(ce)
                i, factor = self._normalize(fi(ce), b)
                b       *= factor
            else:
                b       = fb(ce)
                i       = fi(ce)
            ii += i
            bg += b

        return ce, ii, bg


    def _normalize(self, i, b, area = 1000):
        factor = 1 / np.sum(np.abs(i - b)) * area
        return i * factor, factor


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
        self.calibrations           = [] # List of energy calibrations



    def get_spectrum(self):

        """
        """


        start = time.time()
        active_analyzers = list([a for a in self.analyzers if a.active])
        active_background_rois = list([b for b in self.bg_rois if b.active])
        active_scans = list([s for s in self.scans if s.active])

        if len(active_scans) < 1:
            raise ValueError("No active scans!")
        if len(active_analyzers) < 1:
            raise ValueError("No active analyzers!")

        types = list(
            [(s.name,'{}f4'.format(len(s.images))) for s in active_scans])

        # e = []#np.empty(len(active_analyzers), types)
        # i = []#np.empty(len(active_analyzers), types)
        # b = []#np.empty(len(active_analyzers), types)

        result = AnalysisResult()

        for scan in active_scans:

            index = self.scans.index(scan)

            # Calibrate analyzers
            try:
                calibration = self.calibrations[index]
                calibration.calibrate_energy_for_analyzers(
                    analyzers = active_analyzers)
            except Exception as e:
                calibration = None
                fmt = "Energy calibration for scan {} failed: {}."
                Log.error(fmt.format(scan.name, e))

            in_e, out_e, inte, back, fit = scan.get_energy_spectrum(active_analyzers,
                active_background_rois, calibration)

            d = {scan.name : list([a.name for a in active_analyzers])}
            result.add_data(in_e, out_e, inte, back, fit, d)

        end = time.time()
        fmt = "Single spectra obtained [Took {:2f} s]".format(end-start)
        Log.debug(fmt)
        start = end


        end = time.time()
        fmt = "Spectra summed [Took {:2f} s]".format(end-start)
        Log.debug(fmt)



        return result


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


    def add_scan(self, scan, calibration = None):
        """Add a scan object. Specify scan and corresponding calibration.
        """


        self.scans.append(scan)
        self.calibrations.append(calibration)


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


    def change_calibration(self, scan, calibration):
        """Replace current energy calibration for *scan* with
        *calibration*. Raise an exception, if *scan* is unknown.
        """
        if scan not in self.scans:
            fmt = "Calibration could not be set for scan. Scan {} unknown."
            raise ValueError(fmt.format(scan))

        self.calibrations[self.scans.index(scan)] = calibration




class Scan(object):
    def __init__(self, log_file, image_files):
        """ Specify *logfile* as namestring and list of *image_files*.
        Assumes that logfile holds energy for each image.
        """
        self.name           = os.path.splitext(os.path.split(log_file)[1])[0]
        self.log_file       = log_file
        self.files          = image_files
        self.images         = None
        self.energies       = None
        self.monitor        = None
        self.active         = True
        # self.bg_model     = None
        self.loaded         = False
        self.offset         = [0,0]
        self.range          = [0, len(image_files)]


    @property
    def images(self):
        return self.__images[slice(*self.range)]


    @property
    def energies(self):
        return self.__energies[slice(*self.range)]


    @property
    def monitor(self):
        return self.__monitor[slice(*self.range)]


    @images.setter
    def images(self, images):
        self.__images = images


    @energies.setter
    def energies(self, energies):
        self.__energies = energies


    @monitor.setter
    def monitor(self, monitor):
        self.__monitor = monitor


    def read_logfile(self):
        with open(self.log_file, 'r') as content_file:
            content = content_file.read()

        # pattern = r'(\d+\.\d+)\s'*8+r'.*'+r'(\d+\.\d+)\s'*3

        # Before pin3 diode stopped working
        fio_version = 1
        pattern = r'\s*([+-]*\d+\.*\d*[e0-9-+]*)\s'*17
        matches = re.findall(pattern, content)


        fmt = "Fio file version: {}, files: {}"
        Log.debug(fmt.format(fio_version, len(matches)))



        enc_dcm_ener = [0]*len(matches)
        i01 = np.zeros(len(matches))
        i02 = np.zeros(len(matches))
        tfy = np.zeros(len(matches))
        trans = np.zeros(len(matches))

        for ind, match in enumerate(matches):


            if fio_version == 1:
                # Before pin3 diode stopped working
                e, _, _, i01str, _ ,tfystr, _, _, _, _, _, _, _, _, _, _, _ = match
                enc_dcm_ener[ind] = float(e)
                i01[ind] = float(i01str)
                tfy[ind] = float(tfystr)
                #trans[ind] = float(transstr)

                i0 = i01


        self.energies = enc_dcm_ener
        self.monitor = i0

        # plt.plot(pin2)
        # plt.show()


    def read_files(self, callback = None):

        self.images = np.empty(len(self.energies), dtype = '(195,487)i4')

        # print(self.images, self.files)
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


    def get_energy_spectrum(self, analyzers, background_rois,
        calibration = None):

        in_e = np.array(self.energies)
        out_e = np.empty((len(analyzers)), dtype = list)
        intensity = np.empty((len(analyzers), len(in_e)), dtype = list)
        background = np.empty((len(analyzers), len(in_e)), dtype = list)
        fits = np.empty((len(analyzers)), dtype = list)

        for ind, an in enumerate(analyzers):
            b, s, bg, fit = an.get_signal_series(images = self.images,
                background_rois = background_rois,
                calibration = calibration)

            out_e[ind] = b
            intensity[ind] = s
            background[ind] = bg
            fits[ind] = fit

            # I0
            intensity[ind] /= self.monitor
            background[ind] /= self.monitor

        return in_e, out_e, intensity, background, fits

        # for ind, an in enumerate(analyzers):
        #     b, s = an.get_signal_series(images=self.images)
        #
        #     for ind in range(len(s)):
        #         s[ind] /= self.monitor[ind]
        #
        #     bg = np.zeros(s.shape)
        #     out_e.append(b)
        #     intensity.append(s)
        #     background.append(bg)
        #
        # return in_e, out_e, intensity, background

    #
    # def set_background_model(self, bg_model):
    #     self.bg_model = bg_model


class Analyzer(object):
    def __init__(self, name):
        """
        Analyzers have a
        property called *active*, which can be used to include/exclude them when
        doing summed signal analysis.
        """

        # self.calibration        = None
        self.energy_offset      = 0.0
        self.roi                = None      # xmin,ymin,xmax,ymax e.g. [0,0,5,5]
        self.active             = True
        self.name               = name
        self.mask               = None
        self.poly_fit           = False
        self.poly_order         = 6


    def size(self, mask = None):
        if mask is None:
            x0, y0, x1, y1 = self.roi
        else:
            x0, y0, x1, y1 = self.clip_roi(self.roi, mask)

        size = y1 - y0 + 1
        return size if size > 0 else 0


    def pos(self):
        x0, y0, x1, y1 = self.roi
        return np.array([(y1-y0)/2 + y0, (x1-x0)/2 + x0])


    def set_roi(self, roi):

        if isinstance(roi, list):
            self.roi = np.array(roi)
        elif isinstance(roi, np.ndarray):
            self.roi = roi
        else:
            fmt = "ROI has to be specified like [xmin, ymin, xmax, ymax], "\
                "either as list or np.ndarray."
            raise Exception(fmt)


    def get_roi(self, mask = None):

        if mask is None:
            mask = self.mask

        if mask is None:
            return self.roi
        else:
            return self.clip_roi(self.roi, mask)


    def set_mask(self, mask):
        if isinstance(mask, list):
            self.mask = np.array(mask)
        elif isinstance(mask, np.ndarray):
            self.mask = mask
        else:
            fmt = "Mask has to be specified like [ymax, xmax], "\
                "either as list or np.ndarray."
            raise Exception(fmt)


    def set_calibration(self, calibration):
        self.calibration = calibration.register(self)


    def get_signal(self, image, poly_fit = False, poly_order = 6):
        """
        """

        if self.roi is None:
            raise ValueError("ROI needs to be set before use.")

        x0,y0,x1,y1 = self.clip_roi(self.roi, image.shape)

        ii = np.sum(image[y0:y1+1,x0:x1+1], axis = 0)
        ea = np.arange(len(ii))

        if poly_fit:
            p = np.polyfit(ea, ii, poly_order)
            poly = np.poly1d(p)
            ii = poly(ea)

        return ea, ii


    def get_signal_series(self, images, background_rois = None,
        calibration = None):
        """

        """

        start = time.time()
        x0, y0, x1, y1 = self.clip_roi(self.roi, images[0].shape)

        if calibration is None:
            ea = np.arange(len(np.arange(x0, x1+1)))
            fit = None
        else:
            ea, fit = calibration.get_energy_axis(self)

        ii = np.empty(len(images), dtype = list)
        bg = np.zeros(len(images), dtype = list)


        for ind, image in enumerate(images):
            _, ii[ind] = self.get_signal(image)
            bg[ind] = self.get_background(image, background_rois)



        end = time.time()
        fmt = "Returned signal series [Took {:2f} s]".format(end-start)
        Log.debug(fmt)

        return ea, ii, bg, fit

    def clip_roi(self, roi, shape):
        x0, y0, x1, y1 = roi

        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > shape[1] - 1:
            x1 = shape[1] - 1
        if y1 > shape[0] - 1:
            y1 = shape[0] - 1

        return [x0,y0,x1,y1]


    def get_background(self, image, background_rois):

        bg = np.zeros(image.shape[1])

        upper = None
        lower = None

        # Find nearest background ROIs
        for bg_roi in background_rois:
            if bg_roi.pos()[0] > self.pos()[0]:
                if upper is None:
                    upper = bg_roi
                else:
                    dis_self_roi = np.linalg.norm(self.pos() - bg_roi.pos())
                    dis_self_upper = np.linalg.norm(self.pos() - upper.pos())
                    if dis_self_roi < dis_self_upper:
                        upper = bg_roi
            else:
                if lower is None:
                    lower = bg_roi
                else:
                    dis_self_roi = np.linalg.norm(self.pos() - bg_roi.pos())
                    dis_self_lower = np.linalg.norm(self.pos() - lower.pos())
                    if dis_self_roi < dis_self_lower:
                        lower = bg_roi

        if not upper is None:
            #x0, y0, x1, y1 = self.clip_roi(upper.roi, image.shape)
            #bg_upper = np.sum(image[y0:y1+1, x0:x1+1], axis = 0)
            _, bg_upper = upper.get_signal(image, poly_fit = upper.poly_fit,
                poly_order = upper.poly_order)
            size = upper.size(mask = image.shape)
            if size > 0:
                x0, _, x1, _ = upper.clip_roi(upper.roi, image.shape)
                bg[x0:x1+1] += bg_upper * self.size(mask = image.shape) / size
            else:
                upper = None

        if not lower is None:
            #x0, y0, x1, y1 = self.clip_roi(lower.roi, image.shape)
            #bg_lower = np.sum(image[y0:y1+1, x0:x1+1], axis = 0)
            _, bg_lower = lower.get_signal(image, poly_fit = lower.poly_fit,
                poly_order = lower.poly_order)
            size = lower.size(mask = image.shape)
            if size > 0:
                x0, _, x1, _ = lower.clip_roi(lower.roi, image.shape)
                bg[x0:x1+1] += bg_lower * self.size(mask = image.shape) / size
            else:
                lower = None

        if not lower is None and not upper is None:
            bg /= 2

        x0, _, x1, _ = self.clip_roi(self.roi, image.shape)
        # plt.plot(bg)
        # plt.show()
        return bg[x0:x1+1]






class Calibration(object):
    def __init__(self):
        self.name           = None
        self.elastic_scan   = None
        self.analyzers      = []
        self.calibrations   = []
        self.fits           = []


    def get_energy_axis(self, analyzer):
        if analyzer not in self.analyzers:
            raise Exception("No energy calibration found for this analyzer!")

        x0, _, x1, _ = analyzer.get_roi()
        x = np.arange(x0, x1+1)

        c = self.calibrations[self.analyzers.index(analyzer)](x)
        fit = self.fits[self.analyzers.index(analyzer)]

        return c, fit


    def calibrate_energy_for_analyzers(self, analyzers, elastic_scan = None):

        if elastic_scan is None:
            elastic_scan = self.elastic_scan

        if not isinstance(elastic_scan, Scan):
            raise Exception("Elastic scan needs to be set before calibration!")

        for analyzer in analyzers:
            c, fit = self._calibrate(analyzer, elastic_scan)
            if analyzer in self.analyzers:
                self.calibrations[self.analyzers.index(analyzer)] = c
                self.fits[self.analyzers.index(analyzer)] = fit
            else:
                self.analyzers.append(analyzer)
                self.calibrations.append(c)
                self.fits.append(fit)



    def _calibrate(self, analyzer, elastic_scan, detection_threshold = 0.5):
        mask = elastic_scan.images[0].shape
        x0,y0,x1,y1 = analyzer.get_roi(mask = mask)
        # r = range(*elastic_scan.range)
        images = np.sum(elastic_scan.images[:, y0:y1+1, x0:x1+1], axis = 1)
        threshold = np.max(images[int(len(images) / 2)]) * detection_threshold

        x = []
        y = []
        for ind, image in enumerate(images):
            if np.max(image) < threshold:
                continue

            try:
                pp = self._get_peak_position(np.arange(len(image)), image)
                x.append(elastic_scan.energies[ind])
                y.append(pp)
            except:
                continue

        # Fit
        p = np.poly1d(np.polyfit(y, x, 3))

        # Control
        # plt.plot(y,x, 'ro')
        # plt.plot(y, p(y), 'b')
        # plt.show()

        return lambda x : p(x), [y,x,p(y)]


    def _get_peak_position(self, x, y, epsilon = 5):
        x_max = np.argmax(y)
        x_lower = x_max - epsilon
        x_upper = x_max + epsilon

        if x_lower < 0 or x_upper >= len(y):
            raise Exception("The peak seems to be to close to the edge!")

        x_cutout = np.arange(x_lower, x_upper+1)
        y_cutout = y[x_cutout]

        cumsum = np.cumsum(y_cutout)
        f = interp.interp1d(cumsum, x_cutout)
        return float(f(0.5*np.max(cumsum)))
