from os import makedirs
from os.path import exists, dirname

import numpy as np
from tqdm import tqdm
from tqdm import trange
from traitlets import Dict, List, Int, Unicode, Bool
from matplotlib import pyplot as plt
import pandas as pd

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import AverageWfPeakIntegrator
from ctapipe.image.waveform_cleaning import CHECMWaveformCleanerAverage
from targetpipe.fitting.chec import CHECBrightFitter, SPEFitterFactory
from targetpipe.io.file_looper import TargetioFileLooper
from targetpipe.io.pixels import Dead
from targetpipe.plots.official import ChecmPaperPlotter


class RunsLinePlotter(ChecmPaperPlotter):
    name = 'RunsLinePlotter'

    def __init__(self, config, tool, **kwargs):
        """
        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, tool=tool, **kwargs)

    def create(self, x, y, y_err):

        from IPython import embed
        embed()

        plt.plot(x, y)

        # Normalise histogram
        # norm = np.sum(np.diff(edges) * hist)
        # hist_n = hist/norm
        # fit_n = fit/norm

        # Roll axis for easier plotting
        # nbins = hist_n.size
        # hist_tops = np.insert(hist_n, np.arange(nbins), hist_n, axis=0)
        # edges_tops = np.insert(edges, np.arange(edges.shape[0]),
        #                        edges, axis=0)[1:-1]

        # self.ax.semilogy(edges_tops, hist_tops, color='b', alpha=0.2)
        # self.ax.semilogy(fitx, fit_n, color='r')
        # self.ax.set_ylim(ymin=1e-4)
        # self.ax.set_title("SPE Spectrum, Pixel 1559")
        # self.ax.set_xlabel("Amplitude (ADC)")
        # self.ax.set_ylabel("Probability Density")

        # major_locator = MultipleLocator(50)
        # major_formatter = FormatStrFormatter('%d')
        # minor_locator = MultipleLocator(10)
        # self.ax.xaxis.set_major_locator(major_locator)
        # self.ax.xaxis.set_major_formatter(major_formatter)
        # self.ax.xaxis.set_minor_locator(minor_locator)


class DRExtractor(Tool):
    name = "DRExtractor"
    description = "Loop through files to extract information about the " \
                  "dynamic range of the camera."

    fw_list = List(Int, None, allow_none=True,
                   help='List of the fw setting for each run').tag(config=True)
    output_dir = Unicode(None, allow_none=True,
                         help='Directory to save the figures').tag(config=True)

    aliases = Dict(dict(f='TargetioFileLooper.single_file',
                        N='TargetioFileLooper.max_files',
                        max_events='TargetioFileLooper.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        pe='CameraR1CalibratorFactory.pe_path',
                        fitter='SPEFitterFactory.fitter',
                        O='DRExtractor.output_dir',
                        ))
    flags = Dict(dict())

    classes = List([TargetioFileLooper,
                    CameraR1CalibratorFactory,
                    SPEFitterFactory,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.file_looper = None
        self.r1 = None
        self.dl0 = None
        self.cleaner = None
        self.extractor = None
        self.dl1 = None
        self.dead = None
        self.fitter_bright = None
        self.fitter_spe = None

        self.n_runs = None
        self.n_pixels = None
        self.n_samples = None
        self.n_modules = 32

        self.df = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.file_looper = TargetioFileLooper(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin='targetio', **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)
        self.cleaner = CHECMWaveformCleanerAverage(**kwargs)
        self.extractor = AverageWfPeakIntegrator(**kwargs)
        self.dl0 = CameraDL0Reducer(**kwargs)
        self.dl1 = CameraDL1Calibrator(extractor=self.extractor,
                                       cleaner=self.cleaner,
                                       **kwargs)
        self.dead = Dead()

        self.fitter_bright = CHECBrightFitter(**kwargs)
        fitter_factory = SPEFitterFactory(**kwargs)
        fitter_class = fitter_factory.get_class()
        self.fitter_spe = fitter_class(**kwargs)

        file_reader_list = self.file_looper.file_reader_list
        self.n_runs = len(file_reader_list)
        first_event = file_reader_list[0].get_event(0)
        self.n_pixels = first_event.inst.num_pixels[0]
        self.n_samples = first_event.r0.tel[0].num_samples

        self.fw_list = self.fw_list[:self.file_looper.num_readers]
        assert (self.n_runs == len(self.fw_list))

    def start(self):
        df_list = []

        n_runs = len(self.fw_list)
        # Prepare storage array
        charge_mean = np.ma.zeros((self.n_runs, self.n_pixels))
        charge_mean.mask = np.zeros(charge_mean.shape, dtype=np.bool)
        charge_mean.fill_value = 0
        charge_mean_err = np.ma.copy(charge_mean)
        charge_median = np.ma.copy(charge_mean)
        charge_median_err = np.ma.copy(charge_mean)
        charge_fit = np.ma.copy(charge_mean)
        charge_fit_err = np.ma.copy(charge_mean)

        desc1 = "Looping over runs"
        iterable = enumerate(self.file_looper.file_reader_list)
        for fn, fr in tqdm(iterable, total=self.n_runs, desc=desc1):
            source = fr.read()
            n_events = fr.num_events
            dl1 = np.zeros((n_events, self.n_pixels))

            desc2 = "Extracting charge from events"
            for event in tqdm(source, total=n_events, desc=desc2):
                ev = event.count
                self.r1.calibrate(event)
                self.dl0.reduce(event)
                self.dl1.calibrate(event)

                # Perform CHECM Charge Extraction
                dl1 = event.dl1.tel[0].image[0]

                desc2 = "Characterising pixels"
                for pix in trange(self.n_pixels, desc=desc2):
                    df_list.append(dict(fn=fn, ev=ev, pix=pix, dl1=dl1[pix]))

                # dl1_pixel = dl1[:, pix]
                # if pix in self.dead.dead_pixels:
                #     charge_mean.mask[fn, pix] = True
                #     charge_mean_err.mask[fn, pix] = True
                #     charge_median.mask[fn, pix] = True
                #     charge_median_err.mask[fn, pix] = True
                #     charge_fit.mask[fn, pix] = True
                #     charge_fit_err.mask[fn, pix] = True
                #     continue
                #
                # charge_mean[fn, pix] = np.mean(dl1_pixel)
                # charge_mean_err[fn, pix] = np.std(dl1_pixel)
                # charge_median[fn, pix] = np.median(dl1_pixel)
                # p75 = np.percentile(dl1_pixel, 75)
                # p25 = np.percentile(dl1_pixel, 25)
                # charge_median_err[fn, pix] = p75 - p25
                # if not self.fitter_bright.apply(dl1_pixel):
                #     self.log.warning("FN {} Pixel {} could not be fitted"
                #                      .format(fn, pix))
                #     charge_fit.mask[fn, pix] = True
                #     charge_fit_err.mask[fn, pix] = True
                #     continue
                # charge_fit[fn, pix] = self.fitter_bright.coeff['mean']
                # charge_fit_err[fn, pix] = self.fitter_bright.coeff['stddev']

        self.df = pd.DataFrame(df_list)
        store = pd.HDFStore('/Users/Jason/Downloads/plot_dynamic_range.h5')
        store['df'] = self.df

        store = pd.HDFStore('/Users/Jason/Downloads/plot_dynamic_range.h5')
        self.df = store['df']

        from IPython import embed
        embed()

        self.df.groupby('')

    def finish(self):
        pass
        # # Save figures
        # output_dir = dirname(self.output_path)
        # if not exists(output_dir):
        #     self.log.info("Creating directory: {}".format(output_dir))
        #     makedirs(output_dir)
        #
        # np.savez(self.output_path, charge=self.charge,
        #          charge_error=self.charge_err, rundesc=self.rundesc_list)
        # self.log.info("Numpy array saved to: {}".format(self.output_path))


exe = DRExtractor()
exe.run()
