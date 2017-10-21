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
from scipy.stats import norm
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
from targetpipe.plots.official import ChecmPaperPlotter
from targetpipe.io.pixels import Dead, get_geometry
from targetpipe.calib.camera.filter_wheel import FWCalibrator
from targetpipe.utils.dactov import checm_dac_to_volts

from IPython import embed


class ViolinPlotter(ChecmPaperPlotter):
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


class Dist1D(ChecmPaperPlotter):
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


class ImagePlotter(ChecmPaperPlotter):
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


class WaveformPlotter(ChecmPaperPlotter):
    name = 'WaveformPlotter'

    def add(self, waveform, label):
        self.ax.plot(waveform, label=label)

    def create(self, title, y_label):
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (ns)", fontsize=20)
        self.ax.set_ylabel(y_label, fontsize=20)

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
        self.dummy_event = None
        self.fw_calibrator = None

        self.n_pixels = None
        self.n_samples = None

        self.df_file = None

        self.poi = [1825, 1203]

        self.p_scatter_led = None
        self.p_scatter_led_width = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.fw_calibrator = FWCalibrator(**kwargs)

        dfl = []
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

        self.dummy_event = dfl[0]['reader'].get_event(0)
        telid = list(self.dummy_event.r0.tels_with_data)[0]
        r1 = self.dummy_event.r1.tel[telid].pe_samples[0]
        self.n_pixels, self.n_samples = r1.shape

        script = "checm_paper_led"
        self.p_scatter_led = Scatter(**kwargs, script=script, figure_name="scatter_led", shape='wide')
        self.p_scatter_led_width = Scatter(**kwargs, script=script, figure_name="scatter_led_width", shape='wide')

    def start(self):
        df_list = []

        dead = self.dead.get_pixel_mask()
        kernel = general_gaussian(3, p=1.0, sig=1)
        x_base = np.arange(self.n_samples)
        x_interp = np.linspace(0, self.n_samples - 1, 300)
        ind = np.indices((self.n_pixels, x_interp.size))[1]
        r_ind = ind[:, ::-1]
        ind_x = x_interp[ind]
        r_ind_x = x_interp[r_ind]

        saturation_recovery_file = np.load("/Volumes/gct-jason/plots/checm_paper/checm_paper_recovery/saturation_recovery.npz")
        gradient = saturation_recovery_file['gradient']
        intercept = saturation_recovery_file['intercept']

        desc1 = 'Looping through files'
        n_rows = len(self.df_file.index)
        for index, row in tqdm(self.df_file.iterrows(), total=n_rows, desc=desc1):
            path = row['path']
            reader = row['reader']
            type_ = row['type']
            cal = row['cal']
            level = row['level']

            cal_t = 'Calibrated' if cal else 'Uncalibrated'

            source = reader.read()
            n_events = reader.num_events

            dl1 = np.zeros((n_events, self.n_pixels))
            width = np.zeros((n_events, self.n_pixels))
            low_max = np.zeros((n_events, self.n_pixels), dtype=np.bool)

            desc2 = "Extracting Charge"
            for event in tqdm(source, desc=desc2, total=n_events):
                ev = event.count
                self.dl0.reduce(event)
                self.dl1.calibrate(event)
                dl1[ev] = event.dl1.tel[0].image[0]
                dl0 = event.dl0.tel[0].pe_samples[0]
                cleaned = event.dl1.tel[0].cleaned[0]

                smooth_flat = np.convolve(dl0.ravel(), kernel, "same")
                smoothed = np.reshape(smooth_flat, dl0.shape)
                samples_std = np.std(dl0, axis=1)
                smooth_baseline_std = np.std(smoothed, axis=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    smoothed *= (samples_std / smooth_baseline_std)[:, None]
                    smoothed[~np.isfinite(smoothed)] = 0
                dl0 = smoothed

                f = interpolate.interp1d(x_base, dl0, kind=3, axis=1)
                dl0 = f(x_interp)

                grad = np.gradient(dl0)[1]

                t_max = x_interp[np.argmax(dl0, 1)]
                t_start = t_max - 2
                t_end = t_max + 2
                t_window = (ind_x >= t_start[..., None]) & (ind_x < t_end[..., None])
                t_windowed = np.ma.array(dl0, mask=~t_window)
                t_windowed_ind = np.ma.array(ind_x, mask=~t_window)

                max_ = np.max(dl0, axis=1)
                reversed_ = dl0[:, ::-1]
                peak_time_i = np.ones(dl0.shape) * t_max[:, None]
                mask_before = np.ma.masked_less(ind_x, peak_time_i).mask
                mask_after = np.ma.masked_greater(r_ind_x, peak_time_i).mask
                masked_bef = np.ma.masked_array(dl0, mask_before)
                masked_aft = np.ma.masked_array(reversed_, mask_after)
                pe_width = 20
                d_l = np.diff(np.sign(pe_width - masked_aft))
                d_r = np.diff(np.sign(pe_width - masked_bef))
                t_l = x_interp[r_ind[0, np.argmax(d_l, axis=1) + 1]]
                t_r = x_interp[ind[0, np.argmax(d_r, axis=1) + 1]]
                width[ev] = t_r - t_l
                low_max[ev] = (max_ < pe_width)
                width[ev, low_max[ev]] = 0

            charge_masked = self.dead.mask2d(dl1).compressed()
            charge_camera = np.mean(charge_masked)
            q75, q25 = np.percentile(charge_masked, [75, 25])
            charge_err_top_camera = q75 - charge_camera
            charge_err_bottom_camera = charge_camera - q25

            width = np.ma.masked_array(width, mask=low_max)

            ch = gradient[None, :] * width + intercept[None, :]
            recovered_charge = 10 ** (ch ** 2)

            desc3 = "Aggregate charge per pixel"
            for pix in trange(self.n_pixels, desc=desc3):
                pixel_area = dl1[:, pix]
                pixel_width = width[:, pix]
                pixel_low_max = low_max[:, pix].all()
                pixel_rec_ch = recovered_charge[:, pix]
                if pix in self.dead.dead_pixels:
                    continue

                charge = np.mean(pixel_area)
                q75, q25 = np.percentile(pixel_area, [75, 25])
                charge_err_top = q75 - charge
                charge_err_bottom = charge - q25
                w = np.mean(pixel_width)
                w_err = np.std(pixel_width)
                rec_charge = np.mean(pixel_rec_ch)
                q75, q25 = np.percentile(pixel_rec_ch, [75, 25])
                rec_charge_err_top = q75 - rec_charge
                rec_charge_err_bottom = rec_charge - q25
                df_list.append(dict(type=type_, level=level,
                                    cal=cal, cal_t=cal_t,
                                    pixel=pix, tm=pix//64,
                                    charge=charge,
                                    charge_err_top=charge_err_top,
                                    charge_err_bottom=charge_err_bottom,
                                    charge_camera=charge_camera,
                                    charge_err_top_camera=charge_err_top_camera,
                                    charge_err_bottom_camera=charge_err_bottom_camera,
                                    width=w,
                                    width_err=w_err,
                                    low_max=pixel_low_max,
                                    recovered_charge=rec_charge,
                                    rec_charge_err_top=rec_charge_err_top,
                                    rec_charge_err_bottom=rec_charge_err_bottom))

        df = pd.DataFrame(df_list)
        store = pd.HDFStore('/Users/Jason/Downloads/led.h5')
        store['df'] = df

        store = pd.HDFStore('/Users/Jason/Downloads/led.h5')
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

        df_led = df.loc[(df['type'] == 'LED') & (df['cal'])]

        # Create figures
        self.p_scatter_led.create("LED", "Charge (p.e.)", "LED Distribution")
        self.p_scatter_led.set_y_log()
        output_np = join(self.p_scatter_led.output_dir, "pix{}_dr_led.npz")
        for ip, p in enumerate(self.poi):
            df_pix = df_led.loc[df_led['pixel'] == p]
            x = df_pix['level']
            y = df_pix['charge']
            y_err = [df_pix['charge_err_bottom'], df_pix['charge_err_top']]
            label = "Pixel {}".format(p)
            self.p_scatter_led.add(x, y, None, y_err, label)
            self.log.info("Saving numpy array: {}".format(output_np.format(p)))
            np.savez(output_np.format(p), x=x, y=y, x_err=None, y_err=y_err)
        self.p_scatter_led.add_legend()

        self.p_scatter_led_width.create("Width (ns)", "Charge (p.e.)", "LED Saturation Recovery")
        for ip, p in enumerate(self.poi):
            df_pix = df_led.loc[df_led['pixel'] == p]
            x = df_pix['width']
            y = df_pix['charge']
            x_err = df_pix['width_err']
            y_err = [df_pix['charge_err_bottom'], df_pix['charge_err_top']]
            label = "Pixel {}, Pulse Integration".format(p)
            self.p_scatter_led_width.add(x, y, x_err, y_err, label)
            x = df_pix['width']
            y = df_pix['recovered_charge']
            x_err = df_pix['width_err']
            y_err = [df_pix['rec_charge_err_bottom'], df_pix['rec_charge_err_top']]
            label = "Pixel {}, Saturation Recovery".format(p)
            self.p_scatter_led_width.add(x, y, x_err, y_err, label)
        self.p_scatter_led_width.set_y_log()
        self.p_scatter_led_width.add_legend()

    def finish(self):
        # Save figures
        self.p_scatter_led.save()
        self.p_scatter_led_width.save()


if __name__ == '__main__':
    exe = ADC2PEPlots()
    exe.run()
