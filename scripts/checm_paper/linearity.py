from scipy.signal import general_gaussian

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
from scipy.stats import norm, binned_statistic as bs
from scipy import interpolate

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
from targetpipe.utils.dactov import checm_dac_to_volts

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
        # self.ax.set_title("Distribution Before and After ADC2PE Correction")
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

        # self.ax.set_title("Distribution of Charge Across the Camera")
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

        # self.ax.set_title(title)
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

    def add(self, x, y, x_err=None, y_err=None, label='', c=None):
        if not c:
            c = self.ax._get_lines.get_next_color()
        # no_err = y_err == 0
        # err = ~no_err
        # self.ax.errorbar(x[no_err], y[no_err], fmt='o', mew=0.5, color=c, alpha=0.8, markersize=3, capsize=3)
        (_, caps, _) = self.ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', mew=0.5, color=c, alpha=0.8, markersize=3, capsize=3, label=label)

        for cap in caps:
            cap.set_markeredgewidth(1)

    def create(self, x_label="", y_label="", title=""):
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        # self.fig.suptitle(title)

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
        self.ax.legend(loc=loc, prop={'size': 9})


class WaveformPlotter(OfficialPlotter):
    name = 'WaveformPlotter'

    def add(self, waveform, label):
        self.ax.plot(waveform, label=label)

    def create(self, title, y_label):
        # self.ax.set_title(title)
        self.ax.set_xlabel("Time (ns)")
        self.ax.set_ylabel(y_label)

    def save(self, output_path=None):
        self.ax.legend(loc=2)
        super().save(output_path)


class Profile(OfficialPlotter):
    name = 'Profile'

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        self.ddof = 0
        self.bin_edges = None
        self.n = None
        self.s = None
        self.s2 = None

    @staticmethod
    def sum_squared(array):
        return np.sum(array ** 2)

    @property
    def mean(self):
        n = np.ma.masked_where(self.n == 0, self.n)
        return self.s / n

    @property
    def variance(self):
        n = np.ma.masked_where(self.n == 0, self.n)
        return (n * self.s2 - self.s ** 2) / \
               (n * (n - self.ddof))

    @property
    def stddev(self):
        return np.sqrt(self.variance)

    @property
    def stderr(self):
        n = np.ma.masked_where(self.n == 0, self.n)
        return self.stddev/np.sqrt(n)

    def create(self, x_range, n_xbins, log=False):
        if not log:
            empty, self.bin_edges = np.histogram(None, range=x_range,
                                                 bins=n_xbins)
        else:
            if (x_range[0] <= 0) or (x_range[1] <= 0):
                raise ValueError("X range can only be greater than zero"
                                 " for log bins")
            x_range_log = np.log10(x_range)
            empty, self.bin_edges = np.histogram(np.nan, range=x_range_log,
                                                 bins=n_xbins)
            self.bin_edges = 10 ** self.bin_edges
        self.n = np.zeros(empty.shape)
        self.s = np.zeros(empty.shape)
        self.s2 = np.zeros(empty.shape)

    def add(self, x, y):
        # TODO: Use Welford's Method to avoid issues when mean>>stddev
        x = x.ravel()
        y = y.ravel()
        count, _, _ = bs(x, y, statistic='count', bins=self.bin_edges)
        s, _, _ = bs(x, y, statistic='sum', bins=self.bin_edges)
        s2, _, _ = bs(x, y, statistic=self.sum_squared, bins=self.bin_edges)
        s2[np.isnan(s2)] = 0
        self.n += count
        self.s += s
        self.s2 += s2

    def save_numpy(self, path):
        self.log.info("Saving Profile numpy file: {}".format(path))
        np.savez(path, n=self.n, s=self.s, s2=self.s2)

    def load_numpy(self, path):
        self.log.info("Loading Profile numpy file: {}".format(path))
        file = np.load(path)
        self.n = file['n']
        self.s = file['s']
        self.s2 = file['s2']

    def plot(self, x_label, y_label):
        x = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2
        y = self.mean
        y_err = self.stderr
        (_, caps, _) = self.ax.errorbar(x, y, xerr=None, yerr=y_err, fmt='o',
                                        mew=0.5, color='black',
                                        markersize=3, capsize=3)
        for cap in caps:
            cap.set_markeredgewidth(1)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)

    def set_x_log(self):
        self.ax.set_xscale('log')
        self.ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: '{:g}'.format(x)))

    def set_y_log(self):
        self.ax.set_yscale('log')
        self.ax.get_yaxis().set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))


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
        self.dummy_event = None
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
        self.p_time_res = None
        self.p_time_res_pix = None
        self.p_fwhm_camera = None
        self.p_rt_camera = None
        self.p_fwhm_profile = None
        self.p_rt_profile = None
        self.p_fwhm_pix = None
        self.p_rt_pix = None
        self.p_wf_dict = {}
        self.p_wf_zoom_dict = {}
        self.p_avgwf_dict = {}
        self.p_avgwf_zoom_dict = {}

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

        self.dummy_event = dfl[0]['reader'].get_event(0)
        telid = list(self.dummy_event.r0.tels_with_data)[0]
        r1 = self.dummy_event.r1.tel[telid].pe_samples[0]
        self.n_pixels, self.n_samples = r1.shape

        script = "checm_paper_linearity"
        self.p_comparison = ViolinPlotter(**kwargs, script=script, figure_name="comparison", shape='wide')
        # self.p_dist = Dist1D(**kwargs, script=script, figure_name="3050_distribution", shape='wide')
        self.p_image_saturated = ImagePlotter(**kwargs, script=script, figure_name="image_saturated")
        self.p_scatter_pix = Scatter(**kwargs, script=script, figure_name="scatter_pix")
        self.p_scatter_camera = Scatter(**kwargs, script=script, figure_name="scatter_camera")
        self.p_time_res = Scatter(**kwargs, script=script, figure_name="time_resolution")
        self.p_time_res_pix = Scatter(**kwargs, script=script, figure_name="time_resolution_pix")
        self.p_fwhm_camera = Scatter(**kwargs, script=script, figure_name="fwhm_camera")
        self.p_rt_camera = Scatter(**kwargs, script=script, figure_name="rise_time_camera")
        self.p_fwhm_profile = Profile(**kwargs, script=script, figure_name="fwhm_profile")
        self.p_rt_profile = Profile(**kwargs, script=script, figure_name="rt_profile")
        self.p_fwhm_pix = Scatter(**kwargs, script=script, figure_name="fwhm_pix")
        self.p_rt_pix = Scatter(**kwargs, script=script, figure_name="rise_time_pix")
        for p in self.poi:
            self.p_wf_dict[p] = WaveformPlotter(**kwargs, script=script, figure_name="wfs_pix{}".format(p), shape='wide')
            self.p_wf_zoom_dict[p] = WaveformPlotter(**kwargs, script=script, figure_name="wfs_zoom_pix{}".format(p), shape='wide')
            self.p_avgwf_dict[p] = WaveformPlotter(**kwargs, script=script, figure_name="avgwfs_pix{}".format(p), shape='wide')
            self.p_avgwf_zoom_dict[p] = WaveformPlotter(**kwargs, script=script, figure_name="avgwfs_zoom_pix{}".format(p), shape='wide')

        self.p_fwhm_profile.create([0.1, 1000], 20, True)
        self.p_rt_profile.create([0.1, 1000], 20, True)

    def start(self):
        # df_list = []
        #
        # dead = self.dead.get_pixel_mask()
        # kernel = general_gaussian(3, p=1.0, sig=1)
        # x_base = np.arange(self.n_samples)
        # x_interp = np.linspace(0, self.n_samples - 1, 300)
        # ind = np.indices((self.n_pixels, x_interp.size))[1]
        # r_ind = ind[:, ::-1]
        # ind_x = x_interp[ind]
        # r_ind_x = x_interp[r_ind]
        #
        # saturation_recovery_file = np.load("/Volumes/gct-jason/plots/checm_paper/checm_paper_recovery/saturation_recovery.npz")
        # gradient = saturation_recovery_file['gradient']
        # intercept = saturation_recovery_file['intercept']
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
        #     fwhm = np.zeros((n_events, self.n_pixels))
        #     rise_time = np.zeros((n_events, self.n_pixels))
        #     width = np.zeros((n_events, self.n_pixels))
        #     t0_mask = np.zeros((n_events, self.n_pixels))
        #     wfs = np.zeros((n_events, self.n_pixels, self.n_samples))
        #     low_max = np.zeros((n_events, self.n_pixels), dtype=np.bool)
        #
        #     desc2 = "Extracting Charge"
        #     for event in tqdm(source, desc=desc2, total=n_events):
        #         ev = event.count
        #         self.dl0.reduce(event)
        #         self.dl1.calibrate(event)
        #         dl1[ev] = event.dl1.tel[0].image[0]
        #         dl0 = event.dl0.tel[0].pe_samples[0]
        #         cleaned = event.dl1.tel[0].cleaned[0]
        #
        #         smooth_flat = np.convolve(dl0.ravel(), kernel, "same")
        #         smoothed = np.reshape(smooth_flat, dl0.shape)
        #         samples_std = np.std(dl0, axis=1)
        #         smooth_baseline_std = np.std(smoothed, axis=1)
        #         with np.errstate(divide='ignore', invalid='ignore'):
        #             smoothed *= (samples_std / smooth_baseline_std)[:, None]
        #             smoothed[~np.isfinite(smoothed)] = 0
        #         dl0 = smoothed
        #
        #         f = interpolate.interp1d(x_base, dl0, kind=3, axis=1)
        #         dl0 = f(x_interp)
        #
        #         grad = np.gradient(dl0)[1]
        #
        #         t_max = x_interp[np.argmax(dl0, 1)]
        #         t_start = t_max - 2
        #         t_end = t_max + 2
        #         t_window = (ind_x >= t_start[..., None]) & (ind_x < t_end[..., None])
        #         t_windowed = np.ma.array(dl0, mask=~t_window)
        #         t_windowed_ind = np.ma.array(ind_x, mask=~t_window)
        #
        #         t0[ev] = t_max
        #         t0_grad[ev] = x_interp[np.argmax(grad, 1)]
        #         t0_avg[ev] = np.ma.average(t_windowed_ind, weights=t_windowed, axis=1)
        #
        #         max_ = np.max(dl0, axis=1)
        #         reversed_ = dl0[:, ::-1]
        #         peak_time_i = np.ones(dl0.shape) * t_max[:, None]
        #         mask_before = np.ma.masked_less(ind_x, peak_time_i).mask
        #         mask_after = np.ma.masked_greater(r_ind_x, peak_time_i).mask
        #         masked_bef = np.ma.masked_array(dl0, mask_before)
        #         masked_aft = np.ma.masked_array(reversed_, mask_after)
        #         half_max = max_/2
        #         d_l = np.diff(np.sign(half_max[:, None] - masked_aft))
        #         d_r = np.diff(np.sign(half_max[:, None] - masked_bef))
        #         t_l = x_interp[r_ind[0, np.argmax(d_l, axis=1) + 1]]
        #         t_r = x_interp[ind[0, np.argmax(d_r, axis=1) + 1]]
        #         fwhm[ev] = t_r - t_l
        #         _10percent = 0.1 * max_
        #         _90percent = 0.9 * max_
        #         d10 = np.diff(np.sign(_10percent[:, None] - masked_aft))
        #         d90 = np.diff(np.sign(_90percent[:, None] - masked_aft))
        #         t10 = x_interp[r_ind[0, np.argmax(d10, axis=1) + 1]]
        #         t90 = x_interp[r_ind[0, np.argmax(d90, axis=1) + 1]]
        #         rise_time[ev] = t90 - t10
        #         pe_width = 20
        #         d_l = np.diff(np.sign(pe_width - masked_aft))
        #         d_r = np.diff(np.sign(pe_width - masked_bef))
        #         t_l = x_interp[r_ind[0, np.argmax(d_l, axis=1) + 1]]
        #         t_r = x_interp[ind[0, np.argmax(d_r, axis=1) + 1]]
        #         width[ev] = t_r - t_l
        #         low_max[ev] = (max_ < pe_width)
        #         width[ev, low_max[ev]] = 0
        #
        #         low_pe = dl1[ev] < 0.7
        #         t0_mask[ev] = dead | low_pe
        #
        #         # Shift waveform to match t0 between events
        #         ev_avg = np.mean(cleaned, 0)
        #         peak_time = np.argmax(ev_avg)
        #         pts = peak_time - 50
        #         wf_shift = np.zeros((self.n_pixels, self.n_samples))
        #         if pts >= 0:
        #             wf_shift[:, :cleaned[:, pts:].shape[1]] = cleaned[:, pts:]
        #         else:
        #             wf_shift[:, cleaned[:, pts:].shape[1]:] = cleaned[:, :pts]
        #         wfs[ev] = wf_shift
        #
        #     avgwfs = np.mean(wfs, 0)
        #     self.dummy_event.dl0.tel[0].pe_samples = avgwfs[None, ...]
        #     self.dl1.calibrate(self.dummy_event)
        #     avgwfs_charge = self.dummy_event.dl1.tel[0].image[0]
        #
        #     charge_masked = self.dead.mask2d(dl1).compressed()
        #     charge_camera = np.mean(charge_masked)
        #     q75, q25 = np.percentile(charge_masked, [75, 25])
        #     charge_err_top_camera = q75 - charge_camera
        #     charge_err_bottom_camera = charge_camera - q25
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
        #     fwhm = np.ma.masked_array(fwhm, mask=t0_mask)
        #     fwhm_mean_camera = np.ma.mean(fwhm)
        #     fwhm_std_camera = np.ma.std(fwhm)
        #     rise_time = np.ma.masked_array(rise_time, mask=t0_mask)
        #     rise_time_mean_camera = np.ma.mean(rise_time)
        #     rise_time_std_camera = np.ma.std(rise_time)
        #     width = np.ma.masked_array(width, mask=low_max)
        #
        #     ch = gradient[None, :] * width + intercept[None, :]
        #     with np.errstate(over='ignore'):
        #         recovered_charge = 10 ** (ch ** 2)
        #
        #     if ((type_ == 'LS62') & (level <= 2850)) | \
        #         ((type_ == 'LS64') & (level >= 2450)):
        #         profile_charge = np.ma.masked_array(dl1, mask=t0_mask).compressed()
        #         profile_fwhm = fwhm.compressed()
        #         profile_rt = rise_time.compressed()
        #         self.p_fwhm_profile.add(profile_charge, profile_fwhm)
        #         self.p_rt_profile.add(profile_charge, profile_rt)
        #
        #     desc3 = "Aggregate charge per pixel"
        #     for pix in trange(self.n_pixels, desc=desc3):
        #         pixel_area = dl1[:, pix]
        #         t0_pix = t0_shifted[:, pix]
        #         t0_grad_pix = t0_grad_shifted[:, pix]
        #         t0_avg_pix = t0_avg_shifted[:, pix]
        #         wf = wfs[10, pix]
        #         wf_charge = dl1[10, pix]
        #         avgwf = avgwfs[pix]
        #         avgwf_charge = avgwfs_charge[pix]
        #         pixel_width = width[:, pix]
        #         pixel_low_max = low_max[:, pix].all()
        #         pixel_rec_ch = recovered_charge[:, pix]
        #         pixel_fwhm = fwhm[:, pix]
        #         pixel_rt = rise_time[:, pix]
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
        #         w = np.mean(pixel_width)
        #         w_err = np.std(pixel_width)
        #         rec_charge = np.mean(pixel_rec_ch)
        #         q75, q25 = np.percentile(pixel_rec_ch, [75, 25])
        #         rec_charge_err_top = q75 - rec_charge
        #         rec_charge_err_bottom = rec_charge - q25
        #         fwhm_mean = np.ma.mean(pixel_fwhm)
        #         fwhm_std = np.ma.std(pixel_fwhm)
        #         rt_mean = np.ma.mean(pixel_rt)
        #         rt_std = np.ma.std(pixel_rt)
        #         df_list.append(dict(type=type_, level=level,
        #                             cal=cal, cal_t=cal_t,
        #                             pixel=pix, tm=pix//64,
        #                             charge=charge,
        #                             charge_err_top=charge_err_top,
        #                             charge_err_bottom=charge_err_bottom,
        #                             charge_camera=charge_camera,
        #                             charge_err_top_camera=charge_err_top_camera,
        #                             charge_err_bottom_camera=charge_err_bottom_camera,
        #                             tres_camera=tres_camera,
        #                             tgradres_camera=tgradres_camera,
        #                             tavgres_camera=tavgres_camera,
        #                             tres_camera_n=tres_camera_n,
        #                             tres_pix=tres_pix,
        #                             tgradres_pix=tgradres_pix,
        #                             tavgres_pix=tavgres_pix,
        #                             tres_pix_n=tres_pix_n,
        #                             t0_masked=t0_mask[:, pix].all(),
        #                             fwhm=fwhm_mean,
        #                             fwhm_err=fwhm_std,
        #                             rise_time=rt_mean,
        #                             rise_time_err=rt_std,
        #                             fwhm_mean_camera=fwhm_mean_camera,
        #                             fwhm_std_camera=fwhm_std_camera,
        #                             rise_time_mean_camera=rise_time_mean_camera,
        #                             rise_time_std_camera=rise_time_std_camera,
        #                             width=w,
        #                             width_err=w_err,
        #                             low_max=pixel_low_max,
        #                             recovered_charge=rec_charge,
        #                             rec_charge_err_top=rec_charge_err_top,
        #                             rec_charge_err_bottom=rec_charge_err_bottom,
        #                             wf=wf,
        #                             wf_charge=wf_charge,
        #                             avgwf=avgwf,
        #                             avgwf_charge=avgwf_charge))
        #
        # df = pd.DataFrame(df_list)
        # store = pd.HDFStore('/Users/Jason/Downloads/linearity.h5')
        # store['df'] = df
        #
        # self.p_fwhm_profile.save_numpy('/Users/Jason/Downloads/profile_fwhm.npz')
        # self.p_rt_profile.save_numpy('/Users/Jason/Downloads/profile_rt.npz')
        #
        # store = pd.HDFStore('/Users/Jason/Downloads/linearity.h5')
        # df = store['df']
        #
        # # Scale ADC values to match p.e.
        # type_list = np.unique(df['type'])
        # for t in type_list:
        #     df_t = df.loc[df['type'] == t]
        #     level_list = np.unique(df_t['level'])
        #     for l in level_list:
        #         df_l = df_t.loc[df_t['level'] == l]
        #         median_cal = np.median(df_l.loc[df_l['cal'], 'charge'])
        #         median_uncal = np.median(df_l.loc[~df_l['cal'], 'charge'])
        #         ratio = median_cal / median_uncal
        #         b = (df['type'] == t) & (df['level'] == l) & (~df['cal'])
        #         df.loc[b, 'charge'] *= ratio
        #
        # fw_cal = 2450
        # df_laser = df.loc[(df['type'] == 'LS62') | (df['type'] == 'LS64')]
        # df['illumination'] = 0
        # df['illumination_err'] = 0
        # type_list = np.unique(df_laser['type'])
        # for t in type_list:
        #     df_t = df_laser.loc[df_laser['type'] == t]
        #     pixel_list = np.unique(df_t['pixel'])
        #     for p in tqdm(pixel_list):
        #         df_p = df_t.loc[df_t['pixel'] == p]
        #         cal_entry = (df_p['level'] == fw_cal) & (df_p['cal'])
        #         cal_val = df_p.loc[cal_entry, 'charge'].values
        #         self.fw_calibrator.set_calibration(fw_cal, cal_val)
        #         ill = self.fw_calibrator.get_illumination(df_p['level'])
        #         err = self.fw_calibrator.get_illumination_err(df_p['level'])
        #         b = (df['type'] == t) & (df['pixel'] == p)
        #         df.loc[b, 'illumination'] = ill
        #         df.loc[b, 'illumination_err'] = err
        # store = pd.HDFStore('/Users/Jason/Downloads/linearity.h5')
        # store['df_ill'] = df

        store = pd.HDFStore('/Users/Jason/Downloads/linearity.h5')
        df = store['df_ill']

        self.p_fwhm_profile.load_numpy('/Users/Jason/Downloads/profile_fwhm.npz')
        self.p_rt_profile.load_numpy('/Users/Jason/Downloads/profile_rt.npz')

        # df_lj = df.loc[((df['type'] == 'LS62') &
        #                 (df['illumination'] < 20)) |
        #                ((df['type'] == 'LS64') &
        #                 (df['illumination'] >= 20))]
        df_lj = df.loc[((df['type'] == 'LS62') &
                        (df['level'] <= 2850)) |
                       ((df['type'] == 'LS64') &
                        (df['level'] >= 2450))]
        df_ljc = df_lj.loc[df_lj['cal']]
        df_lju = df_lj.loc[~df_lj['cal']]

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
        output_np = join(self.p_scatter_pix.output_dir, "pix{}_dr.npz")
        for ip, p in enumerate(self.poi):
            df_pix = df_ljc.loc[df_ljc['pixel'] == p]
            x = df_pix['illumination']
            y = df_pix['charge']
            x_err = df_pix['illumination_err']
            y_err = [df_pix['charge_err_bottom'], df_pix['charge_err_top']]
            label = "Pixel {}".format(p)
            self.p_scatter_pix.add(x, y, x_err, y_err, label)
            self.log.info("Saving numpy array: {}".format(output_np.format(p)))
            np.savez(output_np.format(p), x=x, y=y, x_err=x_err, y_err=y_err)
        p = 1825
        df_pix = df_ljc.loc[df_ljc['pixel'] == p]
        x = df_pix['illumination']
        y = df_pix['recovered_charge']
        x_err = df_pix['illumination_err']
        y_err = [df_pix['rec_charge_err_bottom'], df_pix['rec_charge_err_top']]
        label = "Pixel {}, Saturation-Recovered".format(p)
        self.p_scatter_pix.add(x, y, x_err, y_err, label)
        self.p_scatter_pix.add_xy_line()
        self.p_scatter_pix.add_legend()

        df_camera = df_ljc.groupby(['type', 'level'])
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
        x_err = df_mean['illumination_err']
        y_err = df_std['charge']
        self.p_scatter_camera.create("Illumination (p.e.)", "Charge (p.e.)", "Camera Distribution")
        self.p_scatter_camera.set_x_log()
        self.p_scatter_camera.set_y_log()
        self.p_scatter_camera.add(x, y, x_err, y_err, '')
        self.p_scatter_camera.add_xy_line()
        self.p_scatter_camera.add_legend()

        df_camera = df_ljc.groupby(['type', 'level'])
        df_sum = df_camera.apply(np.sum)
        b = df_sum['tm'] == 31652  # fix to ensure statistics
        df_mean = df_camera.apply(np.mean)
        df_mean['a'] = np.arange(df_mean.index.size)
        df_mean.loc[~b, 'a'] = -1
        df_mean = df_mean.groupby('a').apply(np.mean)  # fix to ensure statistics
        x = df_mean['illumination']
        y = df_mean['tres_camera']
        x_err = df_mean['illumination_err']
        y_err = None#np.sqrt(df_mean['tres_camera_n'])
        gt1 = x > 1
        self.p_time_res.create("Illumination (p.e.)", "Time Resolution (ns)", "Camera Timing Resolution")
        self.p_time_res.set_x_log()
        self.p_time_res.set_y_log()
        self.p_time_res.add(x[gt1], y[gt1], x_err[gt1], None, "Peak", 'black')
        # y = df_mean['tgradres_camera']
        # self.p_time_res.add(x, y, y_err, "Grad")
        # gt5 = x > 5
        # y = df_mean['tavgres_camera']
        # self.p_time_res.add(x[gt5], y[gt5], None, "Avg")
        # self.p_time_res.add_xy_line()
        # self.p_time_res.add_legend(1)
        self.p_time_res.ax.get_yaxis().set_minor_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))

        self.p_time_res_pix.create("Illumination (p.e.)", "Time Resolution (ns)", "Pixel Timing Resolution")
        self.p_time_res_pix.set_x_log()
        self.p_time_res_pix.set_y_log()
        for ip, p in enumerate(self.poi):
            df_pix = df_ljc.loc[df_ljc['pixel'] == p]
            x = df_pix['illumination']
            y = df_pix['tres_pix']
            x_err = df_pix['illumination_err']
            gt1 = x > 1
            label = "Pixel {}".format(p)
            self.p_time_res_pix.add(x[gt1], y[gt1], x_err[gt1], None, label)
        # self.p_scatter_pix.add_xy_line()
        self.p_time_res_pix.add_legend(1)

        df_camera = df_ljc.groupby(['type', 'level'])
        df_sum = df_camera.apply(np.sum)
        b = df_sum['tm'] == 31652  # fix to ensure statistics
        df_mean = df_camera.apply(np.mean)
        df_mean['a'] = np.arange(df_mean.index.size)
        df_mean.loc[~b, 'a'] = -1
        df_mean = df_mean.groupby('a').apply(np.mean)  # fix to ensure statistics
        x = df_mean['illumination']
        y = df_mean['fwhm_mean_camera']
        x_err = df_mean['illumination_err']
        y_err = df_mean['fwhm_std_camera']
        self.p_fwhm_camera.create("Illumination (p.e.)", "FWHM (ns)", "Camera FWHM")
        self.p_fwhm_camera.set_x_log()
        # self.p_fwhm_camera.set_y_log()
        self.p_fwhm_camera.add(x, y, x_err, y_err, "Peak", 'black')

        df_camera = df_ljc.groupby(['type', 'level'])
        df_sum = df_camera.apply(np.sum)
        b = df_sum['tm'] == 31652  # fix to ensure statistics
        df_mean = df_camera.apply(np.mean)
        df_mean['a'] = np.arange(df_mean.index.size)
        df_mean.loc[~b, 'a'] = -1
        df_mean = df_mean.groupby('a').apply(np.mean)  # fix to ensure statistics
        x = df_mean['illumination']
        y = df_mean['rise_time_mean_camera']
        x_err = df_mean['illumination_err']
        y_err = df_mean['rise_time_std_camera']
        self.p_rt_camera.create("Illumination (p.e.)", "Rise Time (ns)", "Camera Rise Time")
        self.p_rt_camera.set_x_log()
        # self.p_fwhm_camera.set_y_log()
        self.p_rt_camera.add(x, y, x_err, y_err, "Peak", 'black')

        levels = np.unique(df_lju['level'])
        for p, f in self.p_wf_dict.items():
            title = "Pixel {}".format(p)
            f.create(title, "Amplitude (V)")
            df_pixu = df_lju.loc[df_lju['pixel'] == p]
            df_pixc = df_ljc.loc[df_ljc['pixel'] == p]
            for il, l in enumerate(levels):
                df_lu = df_pixu.loc[df_pixu['level'] == l]
                df_lc = df_pixc.loc[df_pixc['level'] == l]
                wf = checm_dac_to_volts(df_lu['wf'].values[0])
                illumination = df_lu['illumination'].values[0]
                charge = df_lc['wf_charge'].values[0]
                label = "{:.2f} p.e. ({:.2f} p.e.)".format(illumination, charge)
                f.add(wf, label)

        levels = np.unique(df_lju['level'])
        for p, f in self.p_wf_zoom_dict.items():
            title = "Pixel {}".format(p)
            f.create(title, "Amplitude (V)")
            df_pixu = df_lju.loc[df_lju['pixel'] == p]
            df_pixc = df_ljc.loc[df_ljc['pixel'] == p]
            for il, l in enumerate(levels):
                df_lu = df_pixu.loc[df_pixu['level'] == l]
                df_lc = df_pixc.loc[df_pixc['level'] == l]
                wf = checm_dac_to_volts(df_lu['wf'].values[0])
                illumination = df_lu['illumination'].values[0]
                charge = df_lc['wf_charge'].values[0]
                label = "{:.2f} p.e. ({:.2f} p.e.)".format(illumination, charge)
                f.add(wf, label)
                # f.ax.set_ylim((-0.2, 0.2))
                f.ax.set_ylim((-0.005, 0.005))

        levels = np.unique(df_lju['level'])
        for p, f in self.p_avgwf_dict.items():
            title = "Pixel {}".format(p)
            f.create(title, "Amplitude (V)")
            df_pixu = df_lju.loc[df_lju['pixel'] == p]
            df_pixc = df_ljc.loc[df_ljc['pixel'] == p]
            for il, l in enumerate(levels):
                df_lu = df_pixu.loc[df_pixu['level'] == l]
                df_lc = df_pixc.loc[df_pixc['level'] == l]
                wf = checm_dac_to_volts(df_lu['avgwf'].values[0])
                illumination = df_lu['illumination'].values[0]
                charge = df_lc['avgwf_charge'].values[0]
                label = "{:.2f} p.e. ({:.2f} p.e.)".format(illumination, charge)
                f.add(wf, label)

        levels = np.unique(df_lju['level'])
        for p, f in self.p_avgwf_zoom_dict.items():
            title = "Pixel {}".format(p)
            f.create(title, "Amplitude (V)")
            df_pixu = df_lju.loc[df_lju['pixel'] == p]
            df_pixc = df_ljc.loc[df_ljc['pixel'] == p]
            for il, l in enumerate(levels):
                df_lu = df_pixu.loc[df_pixu['level'] == l]
                df_lc = df_pixc.loc[df_pixc['level'] == l]
                wf = checm_dac_to_volts(df_lu['avgwf'].values[0])
                illumination = df_lu['illumination'].values[0]
                charge = df_lc['avgwf_charge'].values[0]
                label = "{:.2f} p.e. ({:.2f} p.e.)".format(illumination, charge)
                f.add(wf, label)
                # f.ax.set_ylim((-0.2, 0.2))
                f.ax.set_ylim((-0.005, 0.005))

        self.p_fwhm_profile.plot("Pulse Area (p.e.)", "FWHM (ns)")
        self.p_fwhm_profile.set_x_log()
        self.p_rt_profile.plot("Pulse Area (p.e.)", "Rise Time (ns)")
        self.p_rt_profile.set_x_log()

        self.p_fwhm_pix.create("Illumination (p.e.)", "FWHM (ns)", "Pixel FWHM")
        df_um = df_ljc.loc[~df['t0_masked']]
        for ip, p in enumerate(self.poi):
            df_pix = df_um.loc[df_um['pixel'] == p]
            x = df_pix['illumination'].values.astype(np.float)
            y = df_pix['fwhm'].values.astype(np.float)
            x_err = df_pix['illumination_err'].values.astype(np.float)
            y_err = df_pix['fwhm_err'].values.astype(np.float)
            label = "Pixel {}".format(p)
            self.p_fwhm_pix.add(x, y, x_err, y_err, label)
        self.p_fwhm_pix.set_x_log()
        self.p_fwhm_pix.add_legend()

        self.p_rt_pix.create("Illumination (p.e.)", "Rise Time (ns)", "Pixel Rise Time")
        df_um = df_ljc.loc[~df['t0_masked']]
        for ip, p in enumerate(self.poi):
            df_pix = df_um.loc[df_um['pixel'] == p]
            x = df_pix['illumination'].values.astype(np.float)
            y = df_pix['fwhm'].values.astype(np.float)
            x_err = df_pix['illumination_err'].values.astype(np.float)
            y_err = df_pix['fwhm_err'].values.astype(np.float)
            label = "Pixel {}".format(p)
            self.p_rt_pix.add(x, y, x_err, y_err, label)
        self.p_rt_pix.set_x_log()
        self.p_rt_pix.add_legend()

    def finish(self):
        # Save figures
        self.p_comparison.save()
        # self.p_dist.save()
        self.p_image_saturated.save()
        self.p_scatter_pix.save()
        self.p_scatter_camera.save()
        self.p_time_res.save()
        self.p_time_res_pix.save()
        self.p_fwhm_camera.save()
        self.p_rt_camera.save()
        for p, f in self.p_wf_dict.items():
            f.save()
        for p, f in self.p_wf_zoom_dict.items():
            f.save()
        for p, f in self.p_avgwf_dict.items():
            f.save()
        for p, f in self.p_avgwf_zoom_dict.items():
            f.save()
        self.p_fwhm_profile.save()
        self.p_rt_profile.save()
        self.p_fwhm_pix.save()
        self.p_rt_pix.save()

if __name__ == '__main__':
    exe = ADC2PEPlots()
    exe.run()
