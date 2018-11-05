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

from os.path import exists, join, dirname, realpath
from os import makedirs

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import AverageWfPeakIntegrator
from ctapipe.image.waveform_cleaning import CHECMWaveformCleanerAverage
from ctapipe.visualization import CameraDisplay
from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.calib.camera.r1 import TargetioR1Calibrator
# from targetpipe.fitting.chec import CHECBrightFitter, CHECMSPEFitter
from targetpipe.calib.camera.adc2pe import TargetioADC2PECalibrator
from targetpipe.plots.official import ChecmPaperPlotter
from targetpipe.io.pixels import Dead, get_geometry
from targetpipe.calib.camera.filter_wheel import FWCalibrator
from targetpipe.utils.dactov import checm_dac_to_volts

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

    def add(self, x, y, x_err=None, y_err=None, label='', c=None, fmt='o', **kwargs):
        if not c:
            c = self.ax._get_lines.get_next_color()
        (_, caps, _) = self.ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt=fmt, mew=1, color=c, alpha=1, markersize=3, capsize=3, elinewidth=0.7, label=label, **kwargs)

        for cap in caps:
            cap.set_markeredgewidth(0.7)

    def add_line(self, x, y, label='', **kwargs):
        self.ax.plot(x, y, label=label, **kwargs)

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
        # self.ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: '{:g}'.format(x)))

    def set_y_log(self):
        self.ax.set_yscale('log')
        # self.ax.get_yaxis().set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))

    def add_legend(self, loc=2, **kwargs):
        self.ax.legend(loc=loc, **kwargs)


class Profile(ChecmPaperPlotter):
    name = 'Profile'

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

    def create(self, x, y, x_range, n_xbins, log=False, x_label="", y_label=""):
        if not log:
            empty, bin_edges = np.histogram(None, range=x_range,
                                            bins=n_xbins)
        else:
            if (x_range[0] <= 0) or (x_range[1] <= 0):
                raise ValueError("X range can only be greater than zero"
                                 " for log bins")
            x_range_log = np.log10(x_range)
            empty, bin_edges = np.histogram(np.nan, range=x_range_log,
                                            bins=n_xbins)
            bin_edges = 10 ** bin_edges

        count, _, _ = bs(x, y, statistic='count', bins=bin_edges)
        mean, _, _ = bs(x, y, statistic='mean', bins=bin_edges)
        stddev, _, _ = bs(x, y, statistic=np.std, bins=bin_edges)
        stderr = stddev/count

        x = (bin_edges[1:] + bin_edges[:-1]) / 2
        y = mean
        y_err = stddev
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

        self.p_fwhm_profile = None
        self.p_rt_profile = None
        self.p_scatter_pix = None
        self.p_tres_pix = None
        self.p_fwhm_pix = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.fw_calibrator = FWCalibrator(**kwargs)

        dfl = []
        base_path_pe = "/Volumes/gct-jason/data/170320/linearity/Run{:05}_r1_pe.tio"
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
        base_path_pe = "/Volumes/gct-jason/data/170319/linearity/linearity/Run{:05}_r1_pe.tio"
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

        script = "checm_paper_linearity_events"
        self.p_fwhm_profile = Profile(**kwargs, script=script, figure_name="fwhm_profile")
        self.p_rt_profile = Profile(**kwargs, script=script, figure_name="rt_profile")
        self.p_scatter_pix = Scatter(**kwargs, script=script, figure_name="scatter_pix")
        self.p_tres_pix = Scatter(**kwargs, script=script, figure_name="tres_pix")
        self.p_fwhm_pix = Scatter(**kwargs, script=script, figure_name="fwhm_pix")

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
        #     peak_height = np.zeros((n_events, self.n_pixels))
        #     peak_time = np.zeros((n_events, self.n_pixels))
        #     peak_time_shifted = np.zeros((n_events, self.n_pixels))
        #     fwhm = np.zeros((n_events, self.n_pixels))
        #     rise_time = np.zeros((n_events, self.n_pixels))
        #     width = np.zeros((n_events, self.n_pixels))
        #
        #     event_num = np.indices((n_events, self.n_pixels))[0]
        #     pixel = np.indices((n_events, self.n_pixels))[1]
        #     type_arr = np.full((n_events, self.n_pixels), type_)
        #     cal_arr = np.full((n_events, self.n_pixels), cal)
        #     level_arr = np.full((n_events, self.n_pixels), level)
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
        #         peak_time[ev] = t_max
        #
        #         max_ = np.max(dl0, axis=1)
        #         peak_height[ev] = max_
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
        #         width[ev, max_ < pe_width] = 0
        #
        #     ch = gradient[None, :] * width + intercept[None, :]
        #     with np.errstate(over='ignore'):
        #         recovered_charge = 10 ** (ch ** 2)
        #     recovered_charge[width==0] = 0
        #
        #     peak_time_shifted = peak_time - peak_time.mean(1)[:, None]
        #
        #     dl1 = self.dead.mask2d(dl1).compressed()
        #     peak_height = self.dead.mask2d(peak_height).compressed()
        #     peak_time = self.dead.mask2d(peak_time).compressed()
        #     peak_time_shifted = self.dead.mask2d(peak_time_shifted).compressed()
        #     fwhm = self.dead.mask2d(fwhm).compressed()
        #     rise_time = self.dead.mask2d(rise_time).compressed()
        #     width = self.dead.mask2d(width).compressed()
        #     recovered_charge = self.dead.mask2d(recovered_charge).compressed()
        #     event_num = self.dead.mask2d(event_num).compressed()
        #     pixel = self.dead.mask2d(pixel).compressed()
        #     type_arr = self.dead.mask2d(type_arr).compressed()
        #     cal_arr = self.dead.mask2d(cal_arr).compressed()
        #     level_arr = self.dead.mask2d(level_arr).compressed()
        #
        #     d_run = dict(type=type_arr,
        #                  cal=cal_arr,
        #                  level=level_arr,
        #                  event=event_num,
        #                  pixel=pixel,
        #                  dl1=dl1,
        #                  peak_height=peak_height,
        #                  peak_time=peak_time,
        #                  peak_time_shifted=peak_time_shifted,
        #                  fwhm=fwhm,
        #                  rise_time=rise_time,
        #                  width=width,
        #                  recovered_charge=recovered_charge
        #                  )
        #     df_run = pd.DataFrame(d_run)
        #     df_list.append(df_run)
        #
        # df = pd.concat(df_list)
        # store = pd.HDFStore('/Users/Jason/Downloads/linearity_events.h5')
        # store['df'] = df
        #
        # store = pd.HDFStore('/Users/Jason/Downloads/linearity_events.h5')
        # df = store['df']
        #
        # df_mean = df.groupby(['type', 'level', 'pixel'], as_index=False).mean()
        # df_mean = df_mean.loc[df_mean['cal']]
        # fw_cal = 2450
        # df_laser = df_mean.loc[(df_mean['type'] == 'LS62') | (df_mean['type'] == 'LS64')]
        # df['illumination'] = 0
        # df['illumination_err'] = 0
        # type_list = np.unique(df_laser['type'])
        # for t in tqdm(type_list):
        #     df_t = df_laser.loc[df_laser['type'] == t]
        #     pixel_list = np.unique(df_t['pixel'])
        #     t_bool = df['type'] == t
        #     for p in tqdm(pixel_list):
        #         df_p = df_t.loc[df_t['pixel'] == p]
        #         cal_entry = df_p['level'] == fw_cal
        #         cal_val = df_p.loc[cal_entry, 'dl1'].values
        #         self.fw_calibrator.set_calibration(fw_cal, cal_val)
        #         b = t_bool & (df['pixel'] == p)
        #         level = df.loc[b, 'level']
        #         ill = self.fw_calibrator.get_illumination(level)
        #         err = self.fw_calibrator.get_illumination_err(level)
        #         data = np.column_stack([ill, err])
        #         df.loc[b, ['illumination', 'illumination_err']] = data
        # store = pd.HDFStore('/Users/Jason/Downloads/linearity_events2.h5')
        # store['df'] = df

        store = pd.HDFStore('/Users/Jason/Downloads/linearity_events2.h5')
        df = store['df']

        df_lj = df.loc[((df['type'] == 'LS62') &
                        (df['level'] <= 2850)) |
                       ((df['type'] == 'LS64') &
                        (df['level'] >= 2450))]
        df_ljc = df_lj.loc[df_lj['cal']]
        df_lju = df_lj.loc[~df_lj['cal']]

        # output = join(self.p_fwhm_profile.output_dir, "data.csv")
        # self.log.info("Saving csv file: {}".format(output))
        # df_ljc.to_csv(output, index=False, index_label=False, float_format='%.6g')

        # Create figures
        df_t = df_ljc.loc[df_ljc['dl1']>0.7]
        x = df_t['dl1']
        y = df_t['fwhm']
        self.p_fwhm_profile.create(x, y, [0.1, 1000], 20, True, "Pixel Area (p.e.)", "FWHM (ns)")
        self.p_fwhm_profile.set_x_log()

        x = df_t['dl1']
        y = df_t['rise_time']
        self.p_rt_profile.create(x, y, [0.1, 1000], 20, True, "Pixel Area (p.e.)", "Rise Time (ns)")
        self.p_rt_profile.set_x_log()

        self.p_scatter_pix.create("Illumination (p.e./pixel)", "Charge (p.e./pixel)", "Pixel Distribution")
        self.p_scatter_pix.set_x_log()
        self.p_scatter_pix.set_y_log()
        mapm_path = join(dirname(realpath(__file__)), "DynRange_MeasRW.txt")
        mapm_data = np.loadtxt(mapm_path, delimiter=',')
        mapm_x = np.log10(mapm_data[:, 1])
        mapm_y = np.log10(mapm_data[:, 2])
        z = np.polyfit(mapm_x, mapm_y, 5)
        p = np.poly1d(z)
        fit_x = np.linspace(np.log10(1), np.log10(5000), 100)
        x = 10**fit_x
        y = 10**p(fit_x)
        label = "MAPM Data"
        self.p_scatter_pix.add_line(x, y, label, color='black')
        df_plot = df_ljc.loc[df_ljc['illumination'] > 1]
        fmt = ['o', 'x']
        for ip, p in enumerate(self.poi):
            df_pix = df_plot.loc[df_plot['pixel'] == p]
            df_gb = df_pix.groupby(['type', 'level'])
            x = df_gb['illumination'].mean().values
            y = df_gb['dl1'].mean().values
            x_err = df_gb['illumination_err'].mean().values
            q25 = df_gb['dl1'].apply(np.percentile, 25).values
            q75 = df_gb['dl1'].apply(np.percentile, 75).values
            y_err_top = q75 - y
            y_err_bottom = y - q25
            y_err = [y_err_bottom, y_err_top]
            label = "Pixel {}".format(p)
            self.p_scatter_pix.add(x, y, x_err, y_err, label, fmt=fmt[ip])
        p = 1825
        df_pix = df_plot.loc[df_plot['pixel'] == p]
        df_pix = df_pix[(df_pix["width"]!=0) & (df_pix["illumination"] > 150)]
        df_gb = df_pix.groupby(['type', 'level'])
        x = df_gb['illumination'].mean().values
        y = df_gb['recovered_charge'].mean().values
        x_err = df_gb['illumination_err'].mean().values
        q25 = df_gb['recovered_charge'].apply(np.percentile, 25).values
        q75 = df_gb['recovered_charge'].apply(np.percentile, 75).values
        y_err_top = q75 - y
        y_err_bottom = y - q25
        y_err = [y_err_bottom, y_err_top]
        label = "Pixel {}, Saturation-Recovered".format(p)
        self.p_scatter_pix.add(x, y, x_err, y_err, label, fmt='v')
        self.p_scatter_pix.add_xy_line()
        self.p_scatter_pix.add_legend(4, markerfirst=False)
        self.p_scatter_pix.ax.set_xlim(left=0.5, right=3000)
        # self.p_scatter_pix.ax.set_ylim(bottom=0.5, top=3000)

        self.p_tres_pix.create("Illumination (p.e./pixel)", "Time Resolution (ns)", "Pixel Timing Resolution")
        self.p_tres_pix.set_x_log()
        self.p_tres_pix.set_y_log()
        df_plot = df_ljc.loc[(df_ljc['illumination'] > 1) & (df_ljc['dl1'] > 0.7)]
        fmt = ['o', 'x']
        for ip, p in enumerate(self.poi):
            df_pix = df_plot.loc[df_plot['pixel'] == p]
            df_gb = df_pix.groupby(['type', 'level'])
            x = df_gb['illumination'].mean().values
            y = df_gb['peak_time_shifted'].std().values
            x_err = df_gb['illumination_err'].mean().values
            label = "Pixel {}".format(p)
            self.p_tres_pix.add(x, y, x_err, None, label, fmt=fmt[ip])
        self.p_tres_pix.add_legend(1)

        self.p_fwhm_pix.create("Peak Height (p.e.)", "FWHM (ns)", "")
        self.p_fwhm_pix.set_x_log()
        df_plot = df_ljc.loc[(df_ljc['fwhm'] > 1) & (df_ljc['illumination'] > 1)]
        # df_plot['peak_height_log'] = np.log10(df_plot['peak_height'])
        marker_mfc = ['black', 'white']
        for ip, p in enumerate(self.poi):
            df_pix = df_plot.loc[df_plot['pixel'] == p]
            x = df_pix['peak_height'].values
            y = df_pix['fwhm'].values
            self.p_fwhm_pix.add(x, y, None, None, None, None, marker=',', zorder=1)
        for ip, p in enumerate(self.poi):
            df_pix = df_plot.loc[df_plot['pixel'] == p]
            df_gb = df_pix.groupby(['type', 'level'])
            x = df_gb['peak_height'].mean().values
            y = df_gb['fwhm'].mean().values
            x_err = df_gb['peak_height'].apply(np.std).values
            y_err = df_gb['fwhm'].apply(np.std).values
            label = "Pixel {}".format(p)
            self.p_fwhm_pix.add(x, y, x_err, y_err, label, 'black', zorder=2, mfc=marker_mfc[ip])
        self.p_fwhm_pix.ax.set_xlim(left=10**-0.8)
        self.p_fwhm_pix.ax.set_ylim([1, 18])
        self.p_fwhm_pix.add_legend()

    def finish(self):
        # Save figures
        self.p_fwhm_profile.save()
        self.p_rt_profile.save()
        self.p_scatter_pix.save()
        self.p_tres_pix.save()
        self.p_fwhm_pix.save()

if __name__ == '__main__':
    exe = ADC2PEPlots()
    exe.run()
