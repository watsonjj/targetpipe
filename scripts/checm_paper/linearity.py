from targetpipe.io.camera import Config
Config('checm')

from tqdm import tqdm, trange
from traitlets import Dict, List
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, \
    AutoMinorLocator, ScalarFormatter, FuncFormatter
import seaborn as sns
from scipy.stats import norm

from os.path import exists, join
from os import makedirs

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import AverageWfPeakIntegrator
from ctapipe.image.waveform_cleaning import CHECMWaveformCleanerAverage
from ctapipe.visualization import CameraDisplay
from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.calib.camera.r1 import TargetioR1Calibrator
from targetpipe.fitting.chec import CHECBrightFitter, CHECMSPEFitter
from targetpipe.calib.camera.adc2pe import TargetioADC2PECalibrator
from targetpipe.plots.official import OfficialPlotter
from targetpipe.io.pixels import Dead, get_geometry
from targetpipe.calib.camera.filter_wheel import FWCalibrator

from IPython import embed


class ViolinPlotter(OfficialPlotter):
    name = 'ADC2PEPlotter'

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

    def create(self, df):
        sns.violinplot(ax=self.ax, data=df, x='level', y='charge', hue='cal_t',
                       split=True, scale='count', inner='quartile',
                       legend=False)
        self.ax.set_title("Distribution Before and After ADC2PE Correction")
        self.ax.set_xlabel('FW')
        self.ax.set_ylabel('Charge (p.e.)')
        self.ax.legend(loc="upper right")

        major_locator = MultipleLocator(50)
        major_formatter = FormatStrFormatter('%d')
        minor_locator = MultipleLocator(10)
        self.ax.yaxis.set_major_locator(major_locator)
        self.ax.yaxis.set_major_formatter(major_formatter)
        self.ax.yaxis.set_minor_locator(minor_locator)


class Dist1D(OfficialPlotter):
    name = 'Dist1D'

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

    def create(self, df):
        charge = df.loc[df['key'] == '3050', 'charge']
        charge_pe = df.loc[df['key'] == '3050pe', 'charge']
        std = charge.std()
        std_pe = charge_pe.std()
        median = np.median(charge)
        median_pe = np.median(charge_pe)
        q75, q25 = np.percentile(charge, [75, 25])
        q75_pe, q25_pe = np.percentile(charge_pe, [75, 25])
        sns.kdeplot(charge, ax=self.ax, color="blue", shade=True, label='Uncal (stddev = {:.2f})'.format(std))
        sns.kdeplot(charge_pe, ax=self.ax, color="green", shade=True, label='Cal, ADC2PE Calibrated (stddev = {:.2f})'.format(std_pe))

        x, y = self.ax.get_lines()[0].get_data()
        y_median_1000 = y[np.abs(x-median).argmin()]
        y_q25_1000 = y[np.abs(x-q25).argmin()]
        y_q75_1000 = y[np.abs(x-q75).argmin()]
        x, y = self.ax.get_lines()[1].get_data()
        y_median_1000pe = y[np.abs(x-median_pe).argmin()]
        y_q25_1000pe = y[np.abs(x-q25_pe).argmin()]
        y_q75_1000pe = y[np.abs(x-q75_pe).argmin()]

        self.ax.vlines(median, 0, y_median_1000, color="blue", linestyle='--')
        self.ax.vlines(q25, 0, y_q25_1000, color="blue", linestyle=':')
        self.ax.vlines(q75, 0, y_q75_1000, color="blue", linestyle=':')
        self.ax.vlines(median_pe, 0, y_median_1000pe, color="green", linestyle='--')
        self.ax.vlines(q25_pe, 0, y_q25_1000pe, color="green", linestyle=':')
        self.ax.vlines(q75_pe, 0, y_q75_1000pe, color="green", linestyle=':')

        self.ax.set_title("Distribution of Charge Across the Camera")
        self.ax.set_xlabel('Charge (p.e.)')
        self.ax.set_ylabel('Density')
        self.ax.legend(loc="upper right", prop={'size': 9})

        majorLocator = MultipleLocator(50)
        majorFormatter = FormatStrFormatter('%d')
        minorLocator = MultipleLocator(10)
        self.ax.xaxis.set_major_locator(majorLocator)
        self.ax.xaxis.set_major_formatter(majorFormatter)
        self.ax.xaxis.set_minor_locator(minorLocator)


class ImagePlotter(OfficialPlotter):
    name = 'ImagePlotter'

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

        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(1, 1, 1)

    def create(self, image, label, title):
        camera = CameraDisplay(get_geometry(), ax=self.ax,
                               image=image,
                               cmap='viridis')
        camera.add_colorbar()
        camera.colorbar.set_label(label, fontsize=20)
        camera.image = image
        camera.colorbar.ax.tick_params(labelsize=30)

        self.ax.set_title(title)
        self.ax.axis('off')


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

    def add(self, x, y, y_err=None, label=''):
        c = self.ax._get_lines.get_next_color()
        # no_err = y_err == 0
        # err = ~no_err
        # self.ax.errorbar(x[no_err], y[no_err], fmt='o', mew=0.5, color=c, alpha=0.8, markersize=3, capsize=3)
        (_, caps, _) = self.ax.errorbar(x, y, yerr=y_err, fmt='o', mew=0.5, color=c, alpha=0.8, markersize=3, capsize=3, label=label)

        for cap in caps:
            cap.set_markeredgewidth(1)

    def create(self, x_label="", y_label="", title=""):
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.fig.suptitle(title)

    def add_xy_line(self):
        lims = [
            np.min([self.ax.get_xlim(), self.ax.get_ylim()]),
            np.max([self.ax.get_xlim(), self.ax.get_ylim()]),
        ]

        # now plot both limits against eachother
        self.ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)
        # self.ax.set_aspect('equal')
        self.ax.set_xlim(lims)
        self.ax.set_ylim(lims)

    def set_x_log(self):
        self.ax.set_xscale('log')
        self.ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: '{:g}'.format(x)))

    def set_y_log(self):
        self.ax.set_yscale('log')
        self.ax.get_yaxis().set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))

    def add_legend(self, loc=2):
        self.ax.legend(loc=loc)

    def save(self, output_path=None):
        super().save(output_path)


class WaveformPlotter(OfficialPlotter):
    name = 'WaveformPlotter'

    def add(self, waveform, label):
        self.ax.plot(waveform, label=label)

    def create(self, title):
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (ns)", fontsize=20)
        self.ax.set_ylabel("Amplitude (p.e.)", fontsize=20)

    def save(self, output_path=None):
        self.ax.legend(loc=2)
        super().save(output_path)


class ADC2PEPlots(Tool):
    name = "ADC2PEPlots"
    description = "Create plots related to adc2pe"

    aliases = Dict(dict(max_events='TargetioFileReader.max_events'
                        ))
    classes = List([TargetioFileReader,
                    TargetioR1Calibrator,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reader_dict = dict()
        self.dl0 = None
        self.dl1 = None
        self.dead = None
        self.fw_calibrator = None

        self.n_pixels = None
        self.n_samples = None

        self.df_file = None

        self.poi = [1825, 1203]

        self.p_comparison = None
        self.p_dist = None
        self.p_image_saturated = None
        self.p_scatter_pix = None
        self.p_scatter_camera = None
        self.p_scatter_led = None
        self.p_time_res = None
        self.p_time_res_pix = None
        self.p_wf_dict = {}
        self.p_avgwf_dict = {}

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.fw_calibrator = FWCalibrator(**kwargs)

        dfl = []
        base_path = "/Volumes/gct-jason/data/170320/linearity/Run{:05}_r1_adc.tio"
        base_path_pe = "/Volumes/gct-jason/data/170320/linearity/Run{:05}_r1_pe.tio"

        dfl.append(dict(path=base_path.format(4160), type="LS64", cal=False, level=1250))
        dfl.append(dict(path=base_path.format(4161), type="LS64", cal=False, level=1450))
        dfl.append(dict(path=base_path.format(4162), type="LS64", cal=False, level=1650))
        dfl.append(dict(path=base_path.format(4163), type="LS64", cal=False, level=1850))
        dfl.append(dict(path=base_path.format(4164), type="LS64", cal=False, level=2050))
        dfl.append(dict(path=base_path.format(4165), type="LS64", cal=False, level=2250))
        dfl.append(dict(path=base_path.format(4166), type="LS64", cal=False, level=2450))
        dfl.append(dict(path=base_path.format(4167), type="LS64", cal=False, level=2650))
        dfl.append(dict(path=base_path.format(4168), type="LS64", cal=False, level=2850))
        dfl.append(dict(path=base_path.format(4169), type="LS64", cal=False, level=3050))

        dfl.append(dict(path=base_path_pe.format(4160), type="LS64", cal=True, level=1250))
        dfl.append(dict(path=base_path_pe.format(4161), type="LS64", cal=True, level=1450))
        dfl.append(dict(path=base_path_pe.format(4162), type="LS64", cal=True, level=1650))
        dfl.append(dict(path=base_path_pe.format(4163), type="LS64", cal=True, level=1850))
        dfl.append(dict(path=base_path_pe.format(4164), type="LS64", cal=True, level=2050))
        dfl.append(dict(path=base_path_pe.format(4165), type="LS64", cal=True, level=2250))
        dfl.append(dict(path=base_path_pe.format(4166), type="LS64", cal=True, level=2450))
        dfl.append(dict(path=base_path_pe.format(4167), type="LS64", cal=True, level=2650))
        dfl.append(dict(path=base_path_pe.format(4168), type="LS64", cal=True, level=2850))
        dfl.append(dict(path=base_path_pe.format(4169), type="LS64", cal=True, level=3050))

        base_path = "/Volumes/gct-jason/data/170319/linearity/linearity/Run{:05}_r1_adc.tio"
        base_path_pe = "/Volumes/gct-jason/data/170319/linearity/linearity/Run{:05}_r1_pe.tio"

        dfl.append(dict(path=base_path.format(3986), type="LS62", cal=False, level=1250))
        dfl.append(dict(path=base_path.format(3987), type="LS62", cal=False, level=1450))
        dfl.append(dict(path=base_path.format(3988), type="LS62", cal=False, level=1650))
        dfl.append(dict(path=base_path.format(3989), type="LS62", cal=False, level=1850))
        dfl.append(dict(path=base_path.format(3990), type="LS62", cal=False, level=2050))
        dfl.append(dict(path=base_path.format(3991), type="LS62", cal=False, level=2250))
        dfl.append(dict(path=base_path.format(3992), type="LS62", cal=False, level=2450))
        dfl.append(dict(path=base_path.format(3993), type="LS62", cal=False, level=2650))
        dfl.append(dict(path=base_path.format(3994), type="LS62", cal=False, level=2850))
        dfl.append(dict(path=base_path.format(3995), type="LS62", cal=False, level=3050))

        dfl.append(dict(path=base_path_pe.format(3986), type="LS62", cal=True, level=1250))
        dfl.append(dict(path=base_path_pe.format(3987), type="LS62", cal=True, level=1450))
        dfl.append(dict(path=base_path_pe.format(3988), type="LS62", cal=True, level=1650))
        dfl.append(dict(path=base_path_pe.format(3989), type="LS62", cal=True, level=1850))
        dfl.append(dict(path=base_path_pe.format(3990), type="LS62", cal=True, level=2050))
        dfl.append(dict(path=base_path_pe.format(3991), type="LS62", cal=True, level=2250))
        dfl.append(dict(path=base_path_pe.format(3992), type="LS62", cal=True, level=2450))
        dfl.append(dict(path=base_path_pe.format(3993), type="LS62", cal=True, level=2650))
        dfl.append(dict(path=base_path_pe.format(3994), type="LS62", cal=True, level=2850))
        dfl.append(dict(path=base_path_pe.format(3995), type="LS62", cal=True, level=3050))

        base_path = "/Volumes/gct-jason/data/170322/led/Run{:05}_r1_adc.tio"
        base_path_pe = "/Volumes/gct-jason/data/170322/led/Run{:05}_r1_pe.tio"

        dfl.append(dict(path=base_path.format(4333), type="LED", cal=False, level=0))
        dfl.append(dict(path=base_path.format(4334), type="LED", cal=False, level=1))
        dfl.append(dict(path=base_path.format(4335), type="LED", cal=False, level=2))
        dfl.append(dict(path=base_path.format(4336), type="LED", cal=False, level=3))
        dfl.append(dict(path=base_path.format(4337), type="LED", cal=False, level=4))
        dfl.append(dict(path=base_path.format(4338), type="LED", cal=False, level=5))
        dfl.append(dict(path=base_path.format(4339), type="LED", cal=False, level=6))
        dfl.append(dict(path=base_path.format(4340), type="LED", cal=False, level=7))
        dfl.append(dict(path=base_path.format(4341), type="LED", cal=False, level=8))
        dfl.append(dict(path=base_path.format(4342), type="LED", cal=False, level=9))
        dfl.append(dict(path=base_path.format(4343), type="LED", cal=False, level=10))
        dfl.append(dict(path=base_path.format(4344), type="LED", cal=False, level=11))
        dfl.append(dict(path=base_path.format(4345), type="LED", cal=False, level=12))
        dfl.append(dict(path=base_path.format(4346), type="LED", cal=False, level=13))
        dfl.append(dict(path=base_path.format(4347), type="LED", cal=False, level=14))
        dfl.append(dict(path=base_path.format(4348), type="LED", cal=False, level=15))
        dfl.append(dict(path=base_path.format(4349), type="LED", cal=False, level=16))
        dfl.append(dict(path=base_path.format(4350), type="LED", cal=False, level=17))
        dfl.append(dict(path=base_path.format(4351), type="LED", cal=False, level=18))
        dfl.append(dict(path=base_path.format(4352), type="LED", cal=False, level=19))
        dfl.append(dict(path=base_path.format(4353), type="LED", cal=False, level=20))
        dfl.append(dict(path=base_path.format(4354), type="LED", cal=False, level=21))
        dfl.append(dict(path=base_path.format(4355), type="LED", cal=False, level=22))
        dfl.append(dict(path=base_path.format(4356), type="LED", cal=False, level=23))
        dfl.append(dict(path=base_path.format(4357), type="LED", cal=False, level=24))
        dfl.append(dict(path=base_path.format(4358), type="LED", cal=False, level=25))
        dfl.append(dict(path=base_path.format(4359), type="LED", cal=False, level=26))
        dfl.append(dict(path=base_path.format(4360), type="LED", cal=False, level=27))
        dfl.append(dict(path=base_path.format(4361), type="LED", cal=False, level=28))
        dfl.append(dict(path=base_path.format(4362), type="LED", cal=False, level=29))
        dfl.append(dict(path=base_path.format(4363), type="LED", cal=False, level=30))
        dfl.append(dict(path=base_path.format(4364), type="LED", cal=False, level=31))
        dfl.append(dict(path=base_path.format(4365), type="LED", cal=False, level=32))
        dfl.append(dict(path=base_path.format(4366), type="LED", cal=False, level=33))
        dfl.append(dict(path=base_path.format(4367), type="LED", cal=False, level=34))
        dfl.append(dict(path=base_path.format(4368), type="LED", cal=False, level=35))
        dfl.append(dict(path=base_path.format(4369), type="LED", cal=False, level=36))
        dfl.append(dict(path=base_path.format(4370), type="LED", cal=False, level=37))
        dfl.append(dict(path=base_path.format(4371), type="LED", cal=False, level=38))
        dfl.append(dict(path=base_path.format(4372), type="LED", cal=False, level=39))

        dfl.append(dict(path=base_path_pe.format(4333), type="LED", cal=True, level=0))
        dfl.append(dict(path=base_path_pe.format(4334), type="LED", cal=True, level=1))
        dfl.append(dict(path=base_path_pe.format(4335), type="LED", cal=True, level=2))
        dfl.append(dict(path=base_path_pe.format(4336), type="LED", cal=True, level=3))
        dfl.append(dict(path=base_path_pe.format(4337), type="LED", cal=True, level=4))
        dfl.append(dict(path=base_path_pe.format(4338), type="LED", cal=True, level=5))
        dfl.append(dict(path=base_path_pe.format(4339), type="LED", cal=True, level=6))
        dfl.append(dict(path=base_path_pe.format(4340), type="LED", cal=True, level=7))
        dfl.append(dict(path=base_path_pe.format(4341), type="LED", cal=True, level=8))
        dfl.append(dict(path=base_path_pe.format(4342), type="LED", cal=True, level=9))
        dfl.append(dict(path=base_path_pe.format(4343), type="LED", cal=True, level=10))
        dfl.append(dict(path=base_path_pe.format(4344), type="LED", cal=True, level=11))
        dfl.append(dict(path=base_path_pe.format(4345), type="LED", cal=True, level=12))
        dfl.append(dict(path=base_path_pe.format(4346), type="LED", cal=True, level=13))
        dfl.append(dict(path=base_path_pe.format(4347), type="LED", cal=True, level=14))
        dfl.append(dict(path=base_path_pe.format(4348), type="LED", cal=True, level=15))
        dfl.append(dict(path=base_path_pe.format(4349), type="LED", cal=True, level=16))
        dfl.append(dict(path=base_path_pe.format(4350), type="LED", cal=True, level=17))
        dfl.append(dict(path=base_path_pe.format(4351), type="LED", cal=True, level=18))
        dfl.append(dict(path=base_path_pe.format(4352), type="LED", cal=True, level=19))
        dfl.append(dict(path=base_path_pe.format(4353), type="LED", cal=True, level=20))
        dfl.append(dict(path=base_path_pe.format(4354), type="LED", cal=True, level=21))
        dfl.append(dict(path=base_path_pe.format(4355), type="LED", cal=True, level=22))
        dfl.append(dict(path=base_path_pe.format(4356), type="LED", cal=True, level=23))
        dfl.append(dict(path=base_path_pe.format(4357), type="LED", cal=True, level=24))
        dfl.append(dict(path=base_path_pe.format(4358), type="LED", cal=True, level=25))
        dfl.append(dict(path=base_path_pe.format(4359), type="LED", cal=True, level=26))
        dfl.append(dict(path=base_path_pe.format(4360), type="LED", cal=True, level=27))
        dfl.append(dict(path=base_path_pe.format(4361), type="LED", cal=True, level=28))
        dfl.append(dict(path=base_path_pe.format(4362), type="LED", cal=True, level=29))
        dfl.append(dict(path=base_path_pe.format(4363), type="LED", cal=True, level=30))
        dfl.append(dict(path=base_path_pe.format(4364), type="LED", cal=True, level=31))
        dfl.append(dict(path=base_path_pe.format(4365), type="LED", cal=True, level=32))
        dfl.append(dict(path=base_path_pe.format(4366), type="LED", cal=True, level=33))
        dfl.append(dict(path=base_path_pe.format(4367), type="LED", cal=True, level=34))
        dfl.append(dict(path=base_path_pe.format(4368), type="LED", cal=True, level=35))
        dfl.append(dict(path=base_path_pe.format(4369), type="LED", cal=True, level=36))
        dfl.append(dict(path=base_path_pe.format(4370), type="LED", cal=True, level=37))
        dfl.append(dict(path=base_path_pe.format(4371), type="LED", cal=True, level=38))
        dfl.append(dict(path=base_path_pe.format(4372), type="LED", cal=True, level=39))

        for d in dfl:
            d['reader'] = TargetioFileReader(input_path=d['path'], **kwargs)
        self.df_file = pd.DataFrame(dfl)

        cleaner = CHECMWaveformCleanerAverage(**kwargs)
        extractor = AverageWfPeakIntegrator(**kwargs)
        self.dl0 = CameraDL0Reducer(**kwargs)
        self.dl1 = CameraDL1Calibrator(extractor=extractor,
                                       cleaner=cleaner,
                                       **kwargs)
        self.dead = Dead()

        first_event = dfl[0]['reader'].get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        r1 = first_event.r1.tel[telid].pe_samples[0]
        self.n_pixels, self.n_samples = r1.shape

        script = "checm_paper_linearity"
        self.p_comparison = ViolinPlotter(**kwargs, script=script, figure_name="comparison", shape='wide')
        # self.p_dist = Dist1D(**kwargs, script=script, figure_name="3050_distribution", shape='wide')
        self.p_image_saturated = ImagePlotter(**kwargs, script=script, figure_name="image_saturated")
        self.p_scatter_pix = Scatter(**kwargs, script=script, figure_name="scatter_pix")
        self.p_scatter_camera = Scatter(**kwargs, script=script, figure_name="scatter_camera")
        self.p_scatter_led = Scatter(**kwargs, script=script, figure_name="scatter_led", shape='wide')
        self.p_time_res = Scatter(**kwargs, script=script, figure_name="time_resolution")
        self.p_time_res_pix = Scatter(**kwargs, script=script, figure_name="time_resolution_pix")
        for p in self.poi:
            self.p_wf_dict[p] = WaveformPlotter(**kwargs, script=script, figure_name="wfs_pix{}".format(p), shape='wide')
            self.p_avgwf_dict[p] = WaveformPlotter(**kwargs, script=script, figure_name="avgwfs_pix{}".format(p), shape='wide')

    def start(self):
        # df_list = []
        #
        # dead = self.dead.get_pixel_mask()
        #
        # desc1 = 'Looping through files'
        # n_rows = len(self.df_file.index)
        # for index, row in tqdm(self.df_file.iterrows(), total=n_rows, desc=desc1):
        #     path = row['path']
        #     reader = row['reader']
        #     type_ = row['type']
        #     cal = row['cal']
        #     level = row['level']
        #
        #     cal_t = 'Calibrated' if cal else 'Uncalibrated'
        #
        #     source = reader.read()
        #     n_events = reader.num_events
        #
        #     dl1 = np.zeros((n_events, self.n_pixels))
        #     t0 = np.zeros((n_events, self.n_pixels))
        #     t0_grad = np.zeros((n_events, self.n_pixels))
        #     t0_avg = np.zeros((n_events, self.n_pixels))
        #     t0_mask = np.zeros((n_events, self.n_pixels))
        #     wfs = np.zeros((n_events, self.n_pixels, self.n_samples))
        #
        #     desc2 = "Extracting Charge"
        #     for event in tqdm(source, desc=desc2, total=n_events):
        #         ev = event.count
        #         self.dl0.reduce(event)
        #         self.dl1.calibrate(event)
        #         dl1[ev] = event.dl1.tel[0].image[0]
        #         dl0 = event.dl0.tel[0].pe_samples[0]
        #         ev_avg = np.mean(dl0, 0)
        #         peak_time = np.argmax(ev_avg)
        #         grad = np.gradient(dl0)[1]
        #
        #         ind = np.indices(dl0.shape)[1]
        #         t_max = np.argmax(dl0, 1)
        #         t_start = t_max - 2
        #         t_end = t_max + 2
        #         t_window = (ind >= t_start[..., None]) & (ind < t_end[..., None])
        #         t_windowed = np.ma.array(dl0, mask=~t_window)
        #         t_windowed_ind = np.ma.array(ind, mask=~t_window)
        #
        #         t0[ev] = t_max
        #         t0_grad[ev] = np.argmax(grad, 1)
        #         t0_avg[ev] = np.ma.average(t_windowed_ind, weights=t_windowed, axis=1)
        #
        #         low_pe = dl1[ev] < 0.7
        #         t0_mask[ev] = dead | low_pe
        #
        #         # Shift waveform to match t0 between events
        #         pts = peak_time - 50
        #         dl0_shift = np.zeros((self.n_pixels, self.n_samples))
        #         if pts >= 0:
        #             dl0_shift[:, :dl0[:, pts:].shape[1]] = dl0[:, pts:]
        #         else:
        #             dl0_shift[:, dl0[:, pts:].shape[1]:] = dl0[:, :pts]
        #         wfs[ev] = dl0_shift
        #
        #     t0_shifted = t0 - t0.mean(1)[:, None]
        #     t0_shifted = np.ma.masked_array(t0_shifted, mask=t0_mask)
        #     t0_grad_shifted = t0_grad - t0_grad.mean(1)[:, None]
        #     t0_grad_shifted = np.ma.masked_array(t0_grad_shifted, mask=t0_mask)
        #     t0_avg_shifted = t0_avg - t0_avg.mean(1)[:, None]
        #     t0_avg_shifted = np.ma.masked_array(t0_avg_shifted, mask=t0_mask)
        #     tres_camera = np.ma.std(t0_shifted)
        #     tgradres_camera = np.ma.std(t0_grad_shifted)
        #     tavgres_camera = np.ma.std(t0_avg_shifted)
        #     tres_camera_n = t0_shifted.count()
        #
        #     desc3 = "Aggregate charge per pixel"
        #     for pix in trange(self.n_pixels, desc=desc3):
        #         pixel_area = dl1[:, pix]
        #         t0_pix = t0_shifted[:, pix]
        #         t0_grad_pix = t0_grad_shifted[:, pix]
        #         t0_avg_pix = t0_avg_shifted[:, pix]
        #         wf = wfs[10, pix]
        #         avgwf = np.mean(wfs[:, pix], 0)
        #         if pix in self.dead.dead_pixels:
        #             continue
        #
        #         charge = np.mean(pixel_area)
        #         q75, q25 = np.percentile(pixel_area, [75, 25])
        #         charge_err_top = q75 - charge
        #         charge_err_bottom = charge - q25
        #         tres_pix = np.ma.std(t0_pix)
        #         tgradres_pix = np.ma.std(t0_grad_pix)
        #         tavgres_pix = np.ma.std(t0_avg_pix)
        #         tres_pix_n = t0_pix.count()
        #         df_list.append(dict(type=type_, level=level,
        #                             cal=cal, cal_t=cal_t,
        #                             pixel=pix, tm=pix//64,
        #                             charge=charge,
        #                             charge_err_top=charge_err_top,
        #                             charge_err_bottom=charge_err_bottom,
        #                             tres_camera=tres_camera,
        #                             tgradres_camera=tgradres_camera,
        #                             tavgres_camera=tavgres_camera,
        #                             tres_camera_n=tres_camera_n,
        #                             tres_pix=tres_pix,
        #                             tgradres_pix=tgradres_pix,
        #                             tavgres_pix=tavgres_pix,
        #                             tres_pix_n=tres_pix_n,
        #                             wf=wf,
        #                             avgwf=avgwf))
        #
        # df = pd.DataFrame(df_list)
        # store = pd.HDFStore('/Users/Jason/Downloads/linearity.h5')
        # store['df'] = df

        store = pd.HDFStore('/Users/Jason/Downloads/linearity.h5')
        df = store['df']

        # Scale ADC values to match p.e.
        type_list = np.unique(df['type'])
        for t in type_list:
            df_t = df.loc[df['type'] == t]
            level_list = np.unique(df_t['level'])
            for l in level_list:
                df_l = df_t.loc[df_t['level'] == l]
                median_cal = np.median(df_l.loc[df_l['cal'], 'charge'])
                median_uncal = np.median(df_l.loc[~df_l['cal'], 'charge'])
                ratio = median_cal / median_uncal
                b = (df['type'] == t) & (df['level'] == l) & (~df['cal'])
                df.loc[b, 'charge'] *= ratio

        df_cal = df.loc[df['cal']]

        # fw_cal = 2450
        # df_laser = df_cal.loc[(df_cal['type'] == 'LS62') | (df_cal['type'] == 'LS64')]
        # df_laser['illumination'] = 0
        # type_list = np.unique(df_laser['type'])
        # for t in type_list:
        #     df_t = df_laser.loc[df_laser['type'] == t]
        #     pixel_list = np.unique(df_t['pixel'])
        #     for p in tqdm(pixel_list):
        #         df_p = df_t.loc[df_t['pixel'] == p]
        #         cal_val = df_p.loc[df_p['level'] == fw_cal, 'charge'].values
        #         self.fw_calibrator.set_calibration(fw_cal, cal_val)
        #         ill = self.fw_calibrator.get_illumination(df_p['level'])
        #         b = (df_laser['type'] == t) & (df_laser['pixel'] == p)
        #         df_laser.loc[b, 'illumination'] = ill
        # store = pd.HDFStore('/Users/Jason/Downloads/linearity.h5')
        # store['df_laser'] = df_laser

        store = pd.HDFStore('/Users/Jason/Downloads/linearity.h5')
        df_laser = store['df_laser']

        df_lj = df_laser.loc[((df_laser['type'] == 'LS62') &
                                 (df_laser['illumination'] < 20)) |
                                ((df_laser['type'] == 'LS64') &
                                 (df_laser['illumination'] >= 20))]
        df_led = df_cal.loc[df_cal['type'] == 'LED']

        # Create figures
        self.p_comparison.create(df.loc[df['type'] == 'LS62'])
        # self.p_dist.create(df)

        image = np.zeros(self.n_pixels)
        b = (df['type'] == 'LS64') & (df['level'] == 3050) & (df['cal'])
        image[df.loc[b, 'pixel']] = df.loc[b, 'charge']
        image = self.dead.mask1d(image)
        self.p_image_saturated.create(image, "Charge (p.e.)", "Saturated Run")

        self.p_scatter_pix.create("Illumination (p.e.)", "Charge (p.e.)", "Pixel Distribution")
        self.p_scatter_pix.set_x_log()
        self.p_scatter_pix.set_y_log()
        for ip, p in enumerate(self.poi):
            df_pix = df_lj.loc[df_lj['pixel'] == p]
            x = df_pix['illumination']
            y = df_pix['charge']
            y_err = [df_pix['charge_err_bottom'], df_pix['charge_err_top']]
            label = "Pixel {}".format(p)
            self.p_scatter_pix.add(x, y, y_err, label)
        self.p_scatter_pix.add_xy_line()
        self.p_scatter_pix.add_legend()

        df_camera = df_lj.groupby(['type', 'level'])
        df_sum = df_camera.apply(np.sum)
        b = df_sum['tm'] == 31652  # fix to ensure statistics
        df_mean = df_camera.apply(np.mean)
        df_mean['a'] = np.arange(df_mean.index.size)
        df_mean.loc[~b, 'a'] = -1
        df_mean = df_mean.groupby('a').apply(np.mean)  # fix to ensure statistics
        df_std = df_camera.apply(np.std)
        df_std['a'] = np.arange(df_std.index.size)
        df_std.loc[~b, 'a'] = -1
        df_std = df_std.groupby('a').apply(np.mean)  # fix to ensure statistics
        x = df_mean['illumination']
        y = df_mean['charge']
        y_err = df_std['charge']
        self.p_scatter_camera.create("Illumination (p.e.)", "Charge (p.e.)", "Camera Distribution")
        self.p_scatter_camera.set_x_log()
        self.p_scatter_camera.set_y_log()
        self.p_scatter_camera.add(x, y, y_err, '')
        self.p_scatter_camera.add_xy_line()
        self.p_scatter_camera.add_legend()

        df_camera = df_lj.groupby(['type', 'level'])
        df_sum = df_camera.apply(np.sum)
        b = df_sum['tm'] == 31652  # fix to ensure statistics
        df_mean = df_camera.apply(np.mean)
        df_mean['a'] = np.arange(df_mean.index.size)
        df_mean.loc[~b, 'a'] = -1
        df_mean = df_mean.groupby('a').apply(np.mean)  # fix to ensure statistics
        x = df_mean['illumination']
        y = df_mean['tres_camera']
        self.p_time_res.create("Illumination (p.e.)", "Time Resolution (ns)", "Camera Timing Resolution")
        self.p_time_res.set_x_log()
        self.p_time_res.set_y_log()
        self.p_time_res.add(x, y, None, "Peak")
        # y = df_mean['tgradres_camera']
        # self.p_time_res.add(x, y, None, "Grad")
        # y = df_mean['tavgres_camera']
        # self.p_time_res.add(x, y, None, "Avg")
        # self.p_time_res.add_xy_line()
        # self.p_time_res.add_legend(1)

        self.p_time_res_pix.create("Illumination (p.e.)", "Time Resolution (ns)", "Pixel Timing Resolution")
        self.p_time_res_pix.set_x_log()
        self.p_time_res_pix.set_y_log()
        for ip, p in enumerate(self.poi):
            df_pix = df_lj.loc[df_lj['pixel'] == p]
            x = df_pix['illumination']
            y = df_pix['tres_pix']
            label = "Pixel {}".format(p)
            self.p_time_res_pix.add(x, y, None, label)
        # self.p_scatter_pix.add_xy_line()
        self.p_time_res_pix.add_legend(1)

        levels = np.unique(df_lj['level'])
        for p, f in self.p_wf_dict.items():
            title = "Pixel {}".format(p)
            f.create(title)
            df_pix = df_lj.loc[df_lj['pixel'] == p]
            for il, l in enumerate(levels):
                df_l = df_pix.loc[df_pix['level'] == l]
                wf = df_l['wf'].values[0]
                illumination = df_l['illumination'].values[0]
                label = "{:.2f} p.e.".format(illumination)
                f.add(wf, label)

        levels = np.unique(df_lj['level'])
        for p, f in self.p_avgwf_dict.items():
            title = "Pixel {}".format(p)
            f.create(title)
            df_pix = df_lj.loc[df_lj['pixel'] == p]
            for il, l in enumerate(levels):
                df_l = df_pix.loc[df_pix['level'] == l]
                wf = df_l['avgwf'].values[0]
                illumination = df_l['illumination'].values[0]
                label = "{:.2f} p.e.".format(illumination)
                f.add(wf, label)

        self.p_scatter_led.create("LED", "Charge (p.e.)", "LED Distribution")
        self.p_scatter_led.set_y_log()
        for ip, p in enumerate(self.poi):
            df_pix = df_led.loc[df_led['pixel'] == p]
            x = df_pix['level']
            y = df_pix['charge']
            y_err = [df_pix['charge_err_bottom'], df_pix['charge_err_top']]
            label = "Pixel {}".format(p)
            self.p_scatter_led.add(x, y, y_err, label)

    def finish(self):
        # Save figures
        self.p_comparison.save()
        # self.p_dist.save()
        self.p_image_saturated.save()
        self.p_scatter_pix.save()
        self.p_scatter_camera.save()
        self.p_scatter_led.save()
        self.p_time_res.save()
        self.p_time_res_pix.save()
        for p, f in self.p_wf_dict.items():
            f.save()
        for p, f in self.p_avgwf_dict.items():
            f.save()


if __name__ == '__main__':
    exe = ADC2PEPlots()
    exe.run()
