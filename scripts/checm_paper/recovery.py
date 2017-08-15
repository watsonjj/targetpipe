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
from scipy import interpolate, stats

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

    def add_line(self, x, y, c=None):
        if not c:
            c = self.ax._get_lines.get_next_color()
        self.ax.plot(x, y, color=c, alpha=0.8)

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
        self.ax.legend(loc=loc)


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

        self.poi = [1825, 1203, 0, 100, 2000, 1500]

        self.p_scatter_pix = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.fw_calibrator = FWCalibrator(**kwargs)

        r1_0320 = TargetioR1Calibrator(pedestal_path='/Volumes/gct-jason/data/170320/pedestal/Run04109_ped.tcal',
                                       tf_path='/Volumes/gct-jason/data/170320/tf/Run04110-04159_tf.tcal',
                                       adc2pe_path='/Users/Jason/Software/CHECAnalysis/targetpipe/adc2pe/adc2pe_1100.tcal',
                                       **kwargs)
        r1_0319 = TargetioR1Calibrator(pedestal_path='/Volumes/gct-jason/data/170319/linearity/pedestal/Run04051_ped.tcal',
                                       tf_path='/Volumes/gct-jason/data/170319/linearity/tf/Run04001-04050_tf.tcal',
                                       adc2pe_path='/Users/Jason/Software/CHECAnalysis/targetpipe/adc2pe/adc2pe_1100.tcal',
                                       **kwargs)

        dfl = []
        base_path = "/Volumes/gct-jason/data/170320/linearity/Run{:05}_r0.tio"
        dfl.append(dict(path=base_path.format(4160), type="LS64", cal=True, level=1250, r1=r1_0320))
        dfl.append(dict(path=base_path.format(4161), type="LS64", cal=True, level=1450, r1=r1_0320))
        dfl.append(dict(path=base_path.format(4162), type="LS64", cal=True, level=1650, r1=r1_0320))
        dfl.append(dict(path=base_path.format(4163), type="LS64", cal=True, level=1850, r1=r1_0320))
        dfl.append(dict(path=base_path.format(4164), type="LS64", cal=True, level=2050, r1=r1_0320))
        dfl.append(dict(path=base_path.format(4165), type="LS64", cal=True, level=2250, r1=r1_0320))
        dfl.append(dict(path=base_path.format(4166), type="LS64", cal=True, level=2450, r1=r1_0320))
        dfl.append(dict(path=base_path.format(4167), type="LS64", cal=True, level=2650, r1=r1_0320))
        dfl.append(dict(path=base_path.format(4168), type="LS64", cal=True, level=2850, r1=r1_0320))
        dfl.append(dict(path=base_path.format(4169), type="LS64", cal=True, level=3050, r1=r1_0320))
        base_path = "/Volumes/gct-jason/data/170319/linearity/linearity/Run{:05}_r0.tio"
        dfl.append(dict(path=base_path.format(3986), type="LS62", cal=True, level=1250, r1=r1_0319))
        dfl.append(dict(path=base_path.format(3987), type="LS62", cal=True, level=1450, r1=r1_0319))
        dfl.append(dict(path=base_path.format(3988), type="LS62", cal=True, level=1650, r1=r1_0319))
        dfl.append(dict(path=base_path.format(3989), type="LS62", cal=True, level=1850, r1=r1_0319))
        dfl.append(dict(path=base_path.format(3990), type="LS62", cal=True, level=2050, r1=r1_0319))
        dfl.append(dict(path=base_path.format(3991), type="LS62", cal=True, level=2250, r1=r1_0319))
        dfl.append(dict(path=base_path.format(3992), type="LS62", cal=True, level=2450, r1=r1_0319))
        dfl.append(dict(path=base_path.format(3993), type="LS62", cal=True, level=2650, r1=r1_0319))
        dfl.append(dict(path=base_path.format(3994), type="LS62", cal=True, level=2850, r1=r1_0319))
        dfl.append(dict(path=base_path.format(3995), type="LS62", cal=True, level=3050, r1=r1_0319))

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

        script = "checm_paper_recovery"
        self.p_scatter_pix = Scatter(**kwargs, script=script, figure_name="recovery_lookup")

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
        # desc1 = 'Looping through files'
        # n_rows = len(self.df_file.index)
        # for index, row in tqdm(self.df_file.iterrows(), total=n_rows, desc=desc1):
        #     path = row['path']
        #     reader = row['reader']
        #     type_ = row['type']
        #     cal = row['cal']
        #     level = row['level']
        #     r1c = row['r1']
        #
        #     cal_t = 'Calibrated' if cal else 'Uncalibrated'
        #
        #     source = reader.read()
        #     n_events = reader.num_events
        #
        #     dl1 = np.zeros((n_events, self.n_pixels))
        #     width = np.zeros((n_events, self.n_pixels))
        #     low_pe = np.zeros((n_events, self.n_pixels), dtype=np.bool)
        #
        #     desc2 = "Extracting Charge"
        #     for event in tqdm(source, desc=desc2, total=n_events):
        #         ev = event.count
        #         r1c.calibrate(event)
        #         self.dl0.reduce(event)
        #         self.dl1.calibrate(event)
        #         r0 = event.r0.tel[0].adc_samples[0]
        #         dl0 = event.dl0.tel[0].pe_samples[0]
        #         dl1[ev] = event.dl1.tel[0].image[0]
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
        #         max_ = np.max(dl0, axis=1)
        #         reversed_ = dl0[:, ::-1]
        #         peak_time_i = np.ones(dl0.shape) * t_max[:, None]
        #         mask_before = np.ma.masked_less(ind_x, peak_time_i).mask
        #         mask_after = np.ma.masked_greater(r_ind_x, peak_time_i).mask
        #         masked_bef = np.ma.masked_array(dl0, mask_before)
        #         masked_aft = np.ma.masked_array(reversed_, mask_after)
        #         pe_width = 20
        #         d_l = np.diff(np.sign(pe_width - masked_aft))
        #         d_r = np.diff(np.sign(pe_width - masked_bef))
        #         t_l = x_interp[r_ind[0, np.argmax(d_l, axis=1) + 1]]
        #         t_r = x_interp[ind[0, np.argmax(d_r, axis=1) + 1]]
        #         width[ev] = t_r - t_l
        #         low_pe[ev] = max_ < pe_width
        #         width[ev, low_pe[ev]] = 0
        #
        #     width = np.ma.masked_array(width, mask=low_pe)
        #
        #     desc3 = "Aggregate charge per pixel"
        #     for pix in trange(self.n_pixels, desc=desc3):
        #         if pix in self.dead.dead_pixels:
        #             continue
        #         pixel_area = dl1[:, pix]
        #         pixel_width = width[:, pix]
        #         pixel_low_pe = low_pe[:, pix].all()
        #
        #         charge = np.mean(pixel_area)
        #         q75, q25 = np.percentile(pixel_area, [75, 25])
        #         charge_err_top = q75 - charge
        #         charge_err_bottom = charge - q25
        #         w = np.mean(pixel_width)
        #         w_err = np.std(pixel_width)
        #
        #         df_list.append(dict(type=type_, level=level,
        #                             cal=cal, cal_t=cal_t,
        #                             pixel=pix, tm=pix//64,
        #                             low_pe=pixel_low_pe,
        #                             charge=charge,
        #                             charge_err_top=charge_err_top,
        #                             charge_err_bottom=charge_err_bottom,
        #                             width=w,
        #                             width_err=w_err,
        #                             pixel_area=pixel_area,
        #                             pixel_width=pixel_width))
        #
        # df = pd.DataFrame(df_list)
        # store = pd.HDFStore('/Users/Jason/Downloads/recovery.h5')
        # store['df'] = df
        #
        store = pd.HDFStore('/Users/Jason/Downloads/recovery.h5')
        df = store['df']

        fw_cal = 2450
        df['illumination'] = 0
        df['illumination_err'] = 0
        type_list = np.unique(df['type'])
        for t in type_list:
            df_t = df.loc[df['type'] == t]
            pixel_list = np.unique(df_t['pixel'])
            for p in tqdm(pixel_list):
                df_p = df_t.loc[df_t['pixel'] == p]
                cal_entry = (df_p['level'] == fw_cal) & (df_p['cal'])
                cal_val = df_p.loc[cal_entry, 'charge'].values
                self.fw_calibrator.set_calibration(fw_cal, cal_val)
                ill = self.fw_calibrator.get_illumination(df_p['level'])
                err = self.fw_calibrator.get_illumination_err(df_p['level'])
                b = (df['type'] == t) & (df['pixel'] == p)
                df.loc[b, 'illumination'] = ill
                df.loc[b, 'illumination_err'] = err
        store = pd.HDFStore('/Users/Jason/Downloads/recovery.h5')
        store['df_ill'] = df

        store = pd.HDFStore('/Users/Jason/Downloads/recovery.h5')
        df = store['df_ill']

        df_lj = df.loc[((df['type'] == 'LS62') &
                        (df['level'] <= 2850)) |
                       ((df['type'] == 'LS64') &
                        (df['level'] >= 2450))]

        gradient = np.zeros(self.n_pixels)
        intercept = np.zeros(self.n_pixels)
        df_um = df_lj.loc[~df['low_pe']]
        for p in np.unique(df_lj['pixel']):
            df_pix = df_um.loc[df_um['pixel'] == p]
            w_arr = []
            i_arr = []
            for index, row in df_pix.iterrows():
                widths = row['pixel_width'].compressed()
                ill = np.array([row['illumination']]*widths.size)
                w_arr.append(widths)
                i_arr.append(ill)
            x = np.concatenate(w_arr)
            y = np.concatenate(i_arr)
            x_sl = x
            y_sl = np.sqrt(np.log10(y))
            m, c, _, _, _ = stats.linregress(x_sl, y_sl)
            gradient[p] = m
            intercept[p] = c
        output_np = join(self.p_scatter_pix.output_dir, "saturation_recovery.npz")
        self.log.info("Saving numpy array: {}".format(output_np))
        np.savez(output_np, gradient=gradient, intercept=intercept)

        # Create figures
        self.p_scatter_pix.create("Width (ns)", "Illumination (p.e.)", "Pixel Distribution")
        df_um = df_lj.loc[~df['low_pe']]
        for ip, p in enumerate(self.poi):
            df_pix = df_um.loc[df_um['pixel'] == p]
            x = df_pix['width'].values.astype(np.float)
            y = df_pix['illumination'].values.astype(np.float)
            x_err = df_pix['width_err'].values.astype(np.float)
            y_err = df_pix['illumination_err'].values.astype(np.float)
            m = gradient[p]
            c = intercept[p]
            x_fit_sl = np.linspace(0, 10, 1000)#np.sqrt(np.log10(np.linspace(100, 1000, 1000)))
            y_fit_sl = m * x_fit_sl + c
            x_fit = x_fit_sl
            y_fit = 10 ** (y_fit_sl**2)
            label = "Pixel {}".format(p)
            c = self.p_scatter_pix.ax._get_lines.get_next_color()
            self.p_scatter_pix.add(x, y, x_err, y_err, label, c)
            self.p_scatter_pix.add_line(x_fit, y_fit, c)
        # self.p_scatter_pix.add_xy_line()
        # self.p_scatter_pix.set_x_log()
        self.p_scatter_pix.set_y_log()
        self.p_scatter_pix.add_legend()

    def finish(self):
        # Save figures
        self.p_scatter_pix.save()


if __name__ == '__main__':
    exe = ADC2PEPlots()
    exe.run()
