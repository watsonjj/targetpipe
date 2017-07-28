from targetpipe.calib.camera.filter_wheel import FWCalibrator
from targetpipe.io.camera import Config
Config('checm')

from tqdm import tqdm, trange
from traitlets import Dict, List
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, \
    ScalarFormatter, FuncFormatter
import seaborn as sns
from os.path import realpath, join, dirname

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import AverageWfPeakIntegrator
from ctapipe.image.waveform_cleaning import CHECMWaveformCleanerAverage
from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.calib.camera.r1 import TargetioR1Calibrator
from targetpipe.fitting.chec import CHECMSPEFitter
from targetpipe.io.pixels import Dead
from targetpipe.calib.camera.adc2pe import TargetioADC2PECalibrator
from targetpipe.plots.official import OfficialPlotter

from IPython import embed


class Scatter(OfficialPlotter):
    name = 'Scatter'

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

        # self.fig = plt.figure(figsize=(12, 8))
        # self.ax = self.fig.add_subplot(1, 1, 1)

    def add(self, x, y, y_err, label):
        c = self.ax._get_lines.get_next_color()
        # no_err = y_err == 0
        # err = ~no_err
        # self.ax.errorbar(x[no_err], y[no_err], fmt='o', mew=0.5, color=c, alpha=0.8, markersize=3, capsize=3)
        (_, caps, _) = self.ax.errorbar(x, y, yerr=y_err, fmt='o', mew=0.5, color=c, alpha=0.8, markersize=3, capsize=3, label=label)

        for cap in caps:
            cap.set_markeredgewidth(1)

    def create(self, x, y, y_err, label, x_label="", y_label="", title=""):
        self.add(x, y, y_err, label)

        # self.ax.set_xscale('log')
        # self.ax.set_yscale('log')
        # self.ax.set_xticks(x)
        # self.ax.get_xaxis().set_major_formatter(ScalarFormatter())
        # self.ax.get_yaxis().set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
        # self.ax.xaxis.set_tick_params(
        #     which='minor',  # both major and minor ticks are affected
        #     bottom='off',  # ticks along the bottom edge are off
        #     top='off',  # ticks along the top edge are off
        #     labelbottom='off')  # labels along the bottom edge are off
        # self.ax.xaxis.set_tick_params(which='major', labelsize=6.5)

        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.fig.suptitle(title)
        # self.ax.xaxis.set_major_locator(AutoMinorLocator(5))
        # self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))

        # axes[1].xaxis.set_minor_locator(AutoMinorLocator(5))
        # axes[2].yaxis.set_minor_locator(AutoMinorLocator(5))

    def save(self, output_path=None):
        # self.ax.legend(loc=2)
        super().save(output_path)


class FWInvestigator(Tool):
    name = "FWInvestigator"
    description = "Investigate the FW"

    aliases = Dict(dict())
    classes = List([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reader = None
        self.dl0 = None
        self.dl1 = None
        self.fitter = None
        self.dead = None
        self.fw_calibrator = None

        directory = join(realpath(dirname(__file__)), "../targetpipe/io")
        self.fw_txt_path = join(directory, "FW.txt")
        self.fw_storage_path = join(directory, "FW.h5")
        self.fw_storage_path_spe = join(directory, "FW_spe_LS62.h5")
        self.spe_fw = 1210

        self.p_attenuation = None
        self.p_pe = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        filepath = '/Volumes/gct-jason/data/170314/spe/Run00073_r1_adc.tio'
        self.reader = TargetioFileReader(input_path=filepath, **kwargs)

        cleaner = CHECMWaveformCleanerAverage(**kwargs)
        extractor = AverageWfPeakIntegrator(**kwargs)
        self.dl0 = CameraDL0Reducer(**kwargs)
        self.dl1 = CameraDL1Calibrator(extractor=extractor,
                                       cleaner=cleaner,
                                       **kwargs)
        self.fitter = CHECMSPEFitter(**kwargs)
        self.fitter.range = [-30, 160]
        self.dead = Dead()
        self.fw_calibrator = FWCalibrator(**kwargs)

        script = "filter_wheel"
        self.p_attenuation = Scatter(**kwargs, script=script, figure_name="attenuation")
        self.p_pe = Scatter(**kwargs, script=script, figure_name="pe")
        # self.p_tmspe = TMSPEFitPlotter(**kwargs, script=script, figure_name="spe_fit_tm24")
        # self.p_tmspe_pe = TMSPEFitPlotter(**kwargs, script=script, figure_name="spe_fit_tm24_pe")
        # self.p_adc2pe = ADC2PEPlotter(**kwargs, script=script, figure_name="adc2pe", shape='wide')
        # self.p_adc2pe_1100tm = ADC2PE1100VTMPlotter(**kwargs, script=script, figure_name="adc2pe_1100V_tms", shape='wide')
        # self.p_adc2pe_1100tm_stats = ADC2PE1100VTMStatsPlotter(**kwargs, script=script, figure_name="adc2pe_1100V_tms_stats", shape='wide')

    def start(self):
        n_events = self.reader.num_events
        first_event = self.reader.get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        n_pixels, n_samples = first_event.r1.tel[telid].pe_samples[0].shape

        dl1 = np.zeros((n_events, n_pixels))
        lambda_ = np.zeros(n_pixels)

        source = self.reader.read()
        desc = "Looping through file"
        for event in tqdm(source, total=n_events, desc=desc):
            index = event.count
            self.dl0.reduce(event)
            self.dl1.calibrate(event)
            dl1[index] = event.dl1.tel[telid].image

        desc = "Fitting pixels"
        for pix in trange(n_pixels, desc=desc):
            if not self.fitter.apply(dl1[:, pix]):
                self.log.warning("Pixel {} couldn't be fit".format(pix))
                continue
            lambda_[pix] = self.fitter.coeff['lambda_']

        lambda_ = self.dead.mask1d(lambda_)
        avg_lamda = np.mean(lambda_)

        self.fw_calibrator.load_from_txt(self.fw_txt_path)
        self.fw_calibrator.save(self.fw_storage_path)
        self.fw_calibrator.set_calibration(self.spe_fw, avg_lamda)
        df = self.fw_calibrator.df

        x = df['position']
        y = df['attenuation_mean']
        y_err = df['attenuation_rms']
        self.p_attenuation.create(x, y, y_err, '', "Postion", "Attenuation", "Filter Wheel Attenuation")

        x = df['position']
        y = df['pe']
        y_err = df['pe_err']
        self.p_pe.create(x, y, y_err, '', "Postion", "Illumination (p.e.)", "Filter Wheel Calibrated")
        self.p_pe.ax.set_yscale('log')
        self.p_pe.ax.get_yaxis().set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))

    def finish(self):
        # Save figures
        self.p_attenuation.save()
        self.p_pe.save()
        self.fw_calibrator.save(self.fw_storage_path_spe)


if __name__ == '__main__':
    exe = FWInvestigator()
    exe.run()
