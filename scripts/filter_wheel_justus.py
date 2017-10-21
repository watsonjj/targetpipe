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
from targetpipe.plots.official import ChecmPaperPlotter

from IPython import embed


class Scatter(ChecmPaperPlotter):
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
        self.ax.set_yscale('log')
        # self.ax.set_xticks(x)
        self.ax.get_xaxis().set_major_formatter(ScalarFormatter())
        self.ax.get_yaxis().set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
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

        self.fw_calibrator = None

        directory = "/Users/Jason/Downloads/quick_analysis_results"
        fw_np_path_y1 = join(directory, "quick_analysis_area_withoutpreamp.npy")
        fw_np_path_y2 = join(directory, "quick_analysis_area_withpreamp-withfilterOD1.npy")
        fw_np_path_y3 = join(directory, "quick_analysis_area_withpreamp.npy")
        fw_np_path_yerr1 = join(directory, "quick_analysis_areaerr_withoutpreamp.npy")
        fw_np_path_yerr2 = join(directory, "quick_analysis_areaerr_withpreamp-withfilterOD1.npy")
        fw_np_path_yerr3 = join(directory, "quick_analysis_areaerr_withpreamp.npy")
        fw_np_path_x1 = join(directory, "quick_analysis_fwpos_withoutpreamp.npy")
        fw_np_path_x2 = join(directory, "quick_analysis_fwpos_withpreamp-withfilterOD1.npy")
        fw_np_path_x3 = join(directory, "quick_analysis_fwpos_withpreamp.npy")

        self.fw_np_y1 = np.load(fw_np_path_y1)
        self.fw_np_y2 = np.load(fw_np_path_y2)
        self.fw_np_y3 = np.load(fw_np_path_y3)
        self.fw_np_yerr1 = np.load(fw_np_path_yerr1)
        self.fw_np_yerr2 = np.load(fw_np_path_yerr2)
        self.fw_np_yerr3 = np.load(fw_np_path_yerr3)
        self.fw_np_x1 = np.load(fw_np_path_x1)
        self.fw_np_x2 = np.load(fw_np_path_x2)
        self.fw_np_x3 = np.load(fw_np_path_x3)

        self.p_attenuation = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.fw_calibrator = FWCalibrator(**kwargs)

        script = "filter_wheel"
        self.p_attenuation = Scatter(**kwargs, script=script, figure_name="attenuation")

    def start(self):

        con = np.concatenate
        df = pd.DataFrame(dict(
            position=con([self.fw_np_x1, self.fw_np_x2, self.fw_np_x3]),
            transmission=con([self.fw_np_y1, self.fw_np_y2, self.fw_np_y3]),
            error=con([self.fw_np_yerr1, self.fw_np_yerr2, self.fw_np_yerr3]),
        ))
        df = df.groupby('position').apply(np.mean)

        self.fw_calibrator.df = df
        self.fw_calibrator.save(self.fw_calibrator.fw_path)

        x = df['position']
        y = df['transmission']
        y_err = df['error']
        self.p_attenuation.create(x, y, y_err, '', "Postion", "Transmission", "Filter Wheel Attenuation")

    def finish(self):
        # Save figures
        self.p_attenuation.save()

if __name__ == '__main__':
    exe = FWInvestigator()
    exe.run()
