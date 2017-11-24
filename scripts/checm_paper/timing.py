from scipy.signal import general_gaussian

from targetpipe.io.camera import Config
Config('checm')

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from traitlets import Dict, List
from scipy.stats import norm
from scipy import interpolate

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.image.charge_extractors import LocalPeakIntegrator
from ctapipe.core import Tool
from ctapipe.visualization import CameraDisplay

from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.calib.camera.r1 import TargetioR1Calibrator
from targetpipe.io.pixels import Dead, get_geometry
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

    def create(self, x, y, x_label="", y_label="", title=""):

        self.ax.plot(x, y, 'x', mew=0.5, alpha=0.4)

        # jp = sns.jointplot(x=x, y=y, stat_func=None)
        # jp.fig.set_figwidth(self.fig.get_figwidth())
        # jp.fig.set_figheight(self.fig.get_figheight())
        # self.fig = jp.fig
        # axes = self.fig.get_axes()
        # self.ax = a xes[0]

        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        # self.fig.suptitle(title)
        self.ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))

        # axes[1].xaxis.set_minor_locator(AutoMinorLocator(5))
        # axes[2].yaxis.set_minor_locator(AutoMinorLocator(5))


class Hist2D(ChecmPaperPlotter):
    name = 'Hist2D'

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

        # self.fig = plt.figure(figsize=(14, 10))
        # self.ax = self.fig.add_subplot(1, 1, 1)

    def create(self, x, x_n, y, y_n, x_label="", y_label="", log=False):
        assert x.shape == y.shape
        x_range = (x.min(), x.max())
        y_range = (y.min(), y.max())
        if x_n > 500:
            x_n = 500
        if y_n > 500:
            y_n = 500
        hist, xedges, yedges = np.histogram2d(x.ravel(), y.ravel(), bins=[x_n, y_n], range=[x_range, y_range])
        hist = np.ma.masked_where(hist == 0, hist)
        z = hist
        norm = None
        if log:
            norm = LogNorm(vmin=hist.min(), vmax=hist.max())
        im = self.ax.pcolormesh(xedges, yedges, z.T, cmap="viridis",
                                edgecolors='white', linewidths=0, norm=norm)
        cbar = self.fig.colorbar(im)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        cbar.set_label("N")

        self.ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))




class WaveformHist1DInt(ChecmPaperPlotter):
    name = 'WaveformHist1DInt'

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

    def add(self, vals, label):
        mean, std = norm.fit(vals)
        fit_x = np.linspace(vals.min()-1, vals.max()+1, 1000)
        fit = norm.pdf(fit_x, mean, std)
        label = "{} (Mean = {:.3}, Stddev = {:.3})".format(label, mean, std)

        n = int(vals.max() - vals.min())
        c = self.ax._get_lines.get_next_color()
        self.ax.hist(vals-0.5, n, color=c, alpha=0.5, rwidth=0.5, label=label)
        # self.ax.plot(fit_x, fit, color=c, alpha=0.8, label=label)

    def create(self, vals, label, title=""):

        self.add(vals, label)

        self.ax.set_xlabel("Time (ns)")
        self.ax.set_ylabel("Counts")
        # self.ax.set_title(title)

        self.ax.xaxis.set_minor_locator(MultipleLocator(1))
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    def save(self, output_path=None):
        self.ax.legend(loc=2)
        super().save(output_path)


class WaveformHist1D(ChecmPaperPlotter):
    name = 'WaveformHist1D'

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

        # self.fig = plt.figure(figsize=(14, 10))
        # self.ax = self.fig.add_subplot(1, 1, 1)

    def add(self, vals, label, n=None):
        mean, std = norm.fit(vals)
        fit_x = np.linspace(vals.min()-1, vals.max()+1, 1000)
        fit = norm.pdf(fit_x, mean, std)
        #label = "{} (Mean = {:.3}, Stddev = {:.3})".format(label, mean, std)

        if not n:
            n = int((vals.max() - vals.min())*1.5+1)
        c = self.ax._get_lines.get_next_color()
        self.ax.hist(vals, n, color=c, alpha=0.5, label=label)#, rwidth=0.5)
        # self.ax.plot(fit_x, fit, color=c, alpha=0.8, label=label)

    def create(self, vals, label, title="", n=None):

        self.add(vals, label, n)

        self.ax.set_xlabel("Time (ns)")
        self.ax.set_ylabel("Counts")
        # self.ax.set_title(title)

        # self.ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        # self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    def save(self, output_path=None):
        self.ax.legend(loc=2, prop={'size': 9})
        super().save(output_path)


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

        # self.ax.set_title(title)
        self.ax.axis('off')


class TimingExtractor(Tool):
    name = "TimingExtractor"
    description = "Loop through a file to extract the timing information"

    aliases = Dict(dict(max_events='TargetioFileReader.max_events',
                        ))
    classes = List([TargetioFileReader,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reader_led = None
        self.reader_laser = None
        self.r1_led = None
        self.r1_laser = None
        self.dl0 = None
        self.dl1 = None
        self.dead = None

        self.df = None

        self.n_events_led = None
        self.n_pixels = None
        self.n_samples = None

        self.p_led_eidvsfci = None
        self.p_led_timevstack = None
        self.p_led_bpvstack = None
        self.p_led_eidvst = None
        self.p_led_eidvstgrad = None
        self.p_led_tvstgrad = None
        self.p_led_tvscharge = None
        self.p_led_tgradvscharge = None
        self.p_led_1deoicomp_wavg = None
        self.p_led_1dcomp_wavg = None
        self.p_led_1deoicomp = None
        self.p_led_1dcomp = None
        self.p_led_imageeoitgrad = None

        self.p_laser_1deoicomp_wavg = None
        self.p_laser_1dcomp_wavg = None
        self.p_laser_1deoicomp = None
        self.p_laser_1dcomp = None
        self.p_laser_1d_final = None
        self.p_laser_1d_final_pix = None
        self.p_laser_imageeoitgrad = None

        self.p_laser_fwhm = None

        self.eoi = 4

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        filepath = '/Volumes/gct-jason/data/170322/led/Run04345_r0.tio'
        self.reader_led = TargetioFileReader(input_path=filepath, **kwargs)
        filepath = '/Volumes/gct-jason/data/170320/linearity/Run04167_r0.tio'
        self.reader_laser = TargetioFileReader(input_path=filepath, **kwargs)

        extractor = LocalPeakIntegrator(**kwargs)

        self.r1_led = TargetioR1Calibrator(pedestal_path='/Volumes/gct-jason/data/170322/pedestal/Run04240_ped.tcal',
                                           tf_path='/Volumes/gct-jason/data/170322/tf/Run04277-04327_tf.tcal',
                                           adc2pe_path='/Users/Jason/Software/CHECAnalysis/targetpipe/adc2pe/adc2pe_800gm_c1.tcal',
                                           **kwargs,
                                           )
        self.r1_laser = TargetioR1Calibrator(pedestal_path='/Volumes/gct-jason/data/170320/pedestal/Run04109_ped.tcal',
                                             tf_path='/Volumes/gct-jason/data/170320/tf/Run04110-04159_tf.tcal',
                                             adc2pe_path='/Users/Jason/Software/CHECAnalysis/targetpipe/adc2pe/adc2pe_1100.tcal',
                                             **kwargs,
                                             )
        self.dl0 = CameraDL0Reducer(**kwargs)
        self.dl1 = CameraDL1Calibrator(extractor=extractor,
                                       **kwargs)

        self.dead = Dead()

        self.n_events_led = self.reader_led.num_events
        first_event = self.reader_led.get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        r1 = first_event.r1.tel[telid].pe_samples[0]
        self.n_pixels, self.n_samples = r1.shape

        p_kwargs = kwargs
        p_kwargs['script'] = "checm_paper_timing"
        p_kwargs['figure_name'] = "led_eid_vs_fci"
        self.p_led_eidvsfci = Scatter(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "led_time_vs_tack"
        self.p_led_timevstack = Scatter(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "led_bp_vs_tack"
        self.p_led_bpvstack = Hist2D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "led_eid_vs_t"
        self.p_led_eidvst = Hist2D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "led_eid_vs_tgrad"
        self.p_led_eidvstgrad = Hist2D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "led_t_vs_tgrad"
        self.p_led_tvstgrad = Hist2D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "led_t_vs_charge"
        self.p_led_tvscharge = Hist2D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "led_tgrad_vs_charge"
        self.p_led_tgradvscharge = Hist2D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "led_1D_comparison_eid{}_wavg".format(self.eoi)
        self.p_led_1deoicomp_wavg = WaveformHist1D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "led_1D_comparison_allevents_wavg".format(self.eoi)
        self.p_led_1dcomp_wavg = WaveformHist1D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "led_1D_comparison_eid{}".format(self.eoi)
        self.p_led_1deoicomp = WaveformHist1DInt(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "led_1D_comparison_allevents".format(self.eoi)
        self.p_led_1dcomp = WaveformHist1D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "led_image_tgrad_eid{}".format(self.eoi)
        self.p_led_imageeoitgrad = ImagePlotter(**p_kwargs)

        p_kwargs['figure_name'] = "laser_1D_comparison_eid{}_wavg".format(self.eoi)
        self.p_laser_1deoicomp_wavg = WaveformHist1D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "laser_1D_comparison_allevents_wavg".format(self.eoi)
        self.p_laser_1dcomp_wavg = WaveformHist1D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "laser_1D_comparison_eid{}".format(self.eoi)
        self.p_laser_1deoicomp = WaveformHist1DInt(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "laser_1D_comparison_allevents".format(self.eoi)
        self.p_laser_1dcomp = WaveformHist1D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "laser_1D_finalmethod"
        self.p_laser_1d_final = WaveformHist1D(**p_kwargs, shape='square')
        p_kwargs['figure_name'] = "laser_1D_finalmethod_pix"
        self.p_laser_1d_final_pix = WaveformHist1D(**p_kwargs, shape='square')
        p_kwargs['figure_name'] = "laser_image_tgrad_eid{}".format(self.eoi)
        self.p_laser_imageeoitgrad = ImagePlotter(**p_kwargs)

        p_kwargs['figure_name'] = "laser_fwhm_allevents"
        self.p_laser_fwhm = WaveformHist1DInt(**p_kwargs, shape='wide')

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
        # readers = [self.reader_led, self.reader_laser]
        # r1s = [self.r1_led, self.r1_laser]
        # run_type = ['led', 'laser']
        #
        # for reader, r1_cal, rt in zip(readers, r1s, run_type):
        #     run = reader.filename
        #     n_events = reader.num_events
        #     source = reader.read()
        #     desc = "Processing Events for {} run".format(rt)
        #     for event in tqdm(source, total=n_events, desc=desc):
        #         ev = event.count
        #         event_id = event.r0.event_id
        #         time = event.trig.gps_time.value
        #         tack = event.meta['tack']
        #         fci = np.copy(event.r0.tel[0].first_cell_ids)
        #         bp = event.r0.tel[0].blockphase
        #
        #         r1_cal.calibrate(event)
        #         self.dl0.reduce(event)
        #         self.dl1.calibrate(event)
        #         r0 = event.r0.tel[0].adc_samples[0]
        #         r1 = event.r1.tel[0].pe_samples[0]
        #         dl1 = event.dl1.tel[0].image[0]
        #
        #         smooth_flat = np.convolve(r1.ravel(), kernel, "same")
        #         smoothed = np.reshape(smooth_flat, r1.shape)
        #         samples_std = np.std(r1, axis=1)
        #         smooth_baseline_std = np.std(smoothed, axis=1)
        #         with np.errstate(divide='ignore', invalid='ignore'):
        #             smoothed *= (samples_std / smooth_baseline_std)[:, None]
        #             smoothed[~np.isfinite(smoothed)] = 0
        #         r1 = smoothed
        #
        #         f = interpolate.interp1d(x_base, r1, kind=3, axis=1)
        #         r1 = f(x_interp)
        #
        #         grad = np.gradient(r1)[1]
        #
        #         saturated = np.any(r0 < 10, 1)
        #         low_pe = np.all(r1 < 10, 1)
        #         mask = dead | saturated | low_pe
        #
        #         t_max = x_interp[np.argmax(r1, 1)]
        #         t_start = t_max - 2
        #         t_end = t_max + 2
        #         t_window = (ind_x >= t_start[..., None]) & (ind_x < t_end[..., None])
        #         t_windowed = np.ma.array(r1, mask=~t_window)
        #         t_windowed_ind = np.ma.array(ind_x, mask=~t_window)
        #         t_avg = np.ma.average(t_windowed_ind, weights=t_windowed, axis=1)
        #
        #         t_grad_max = x_interp[np.argmax(grad, 1)]
        #         t_grad_start = t_grad_max - 2
        #         t_grad_end = t_grad_max + 2
        #         t_grad_window = (ind_x >= t_grad_start[..., None]) & (ind_x < t_grad_end[..., None])
        #         t_grad_windowed = np.ma.array(grad, mask=~t_grad_window)
        #         t_grad_windowed_ind = np.ma.array(ind_x, mask=~t_grad_window)
        #         t_grad_avg = np.ma.average(t_grad_windowed_ind, weights=t_grad_windowed, axis=1)
        #
        #         max_ = np.max(r1, axis=1)
        #         reversed_ = r1[:, ::-1]
        #         peak_time_i = np.ones(r1.shape) * t_max[:, None]
        #         mask_before = np.ma.masked_less(ind_x, peak_time_i).mask
        #         mask_after = np.ma.masked_greater(r_ind_x, peak_time_i).mask
        #         masked_bef = np.ma.masked_array(r1, mask_before)
        #         masked_aft = np.ma.masked_array(reversed_, mask_after)
        #         half_max = max_/2
        #         d_l = np.diff(np.sign(half_max[:, None] - masked_aft))
        #         d_r = np.diff(np.sign(half_max[:, None] - masked_bef))
        #         t_l = x_interp[r_ind[0, np.argmax(d_l, axis=1) + 1]]
        #         t_r = x_interp[ind[0, np.argmax(d_r, axis=1) + 1]]
        #         fwhm = t_r - t_l
        #
        #         # if (t_grad > 60).any():
        #         #     print(event_id)
        #         #
        #         if event_id == 23:
        #             continue
        #
        #         d = dict(run=run,
        #                  type=rt,
        #                  index=ev,
        #                  id=event_id,
        #                  time=time,
        #                  tack=tack,
        #                  fci=fci,
        #                  bp=bp,
        #                  mask=mask,
        #                  dl1=dl1,
        #                  t=t_max,
        #                  t_grad=t_grad_max,
        #                  t_avg=t_avg,
        #                  t_grad_avg=t_grad_avg,
        #                  fwhm=fwhm
        #                  )
        #         df_list.append(d)
        #
        # self.df = pd.DataFrame(df_list)
        # store = pd.HDFStore('/Volumes/gct-jason/plots/checm_paper/df/timing.h5')
        # store['df'] = self.df

        store = pd.HDFStore('/Volumes/gct-jason/plots/checm_paper/df/timing.h5')
        self.df = store['df']

    def finish(self):
        # LED DATA
        df = self.df.loc[self.df['type'] == 'led']

        eid = df['id']
        time = df['time']
        tack = df['tack']

        fci = np.ma.vstack(df['fci'])
        bp = np.ma.vstack(df['bp'])
        dl1 = np.ma.vstack(df['dl1'])
        t_avg = np.ma.vstack(df['t_avg'])
        t_grad_avg = np.ma.vstack(df['t_grad_avg'])
        t = np.ma.vstack(df['t'])
        t_grad = np.ma.vstack(df['t_grad'])
        mask = np.vstack(df['mask'])

        # bp.mask = mask
        dl1.mask = mask
        t_avg.mask = mask
        t_grad_avg.mask = mask
        t.mask = mask
        t_grad.mask = mask

        # Scatter
        self.p_led_eidvsfci.create(eid, fci, 'Event ID', 'First Cell ID')
        self.p_led_timevstack.create(time, tack, 'Time', 'Tack')

        # 2D histograms
        eid_pix = eid[:, None] * np.ma.ones((eid.size, self.n_pixels))
        eid_pix.mask = mask
        eid_pix_c = eid_pix.compressed()
        tack_pix = tack[:, None] * np.ma.ones((tack.size, self.n_pixels))
        # tack_pix.mask = mask
        tack_pix_c = tack_pix.compressed()
        bp_c = bp.compressed()
        dl1_c = dl1.compressed()
        t_c = t_avg.compressed()
        t_grad_c = t_grad_avg.compressed()
        n_bp = int(bp_c.max() - bp_c.min())
        n_t = int(t_c.max() - t_c.min())
        n_tgrad = int(t_grad_c.max() - t_grad_c.min())
        self.p_led_bpvstack.create(bp_c, n_bp, tack_pix_c, self.n_events_led, 'Blockphase', 'Tack')
        self.p_led_eidvst.create(eid_pix_c, self.n_events_led, t_c, n_t, 'Event ID', 'Peak Time')
        self.p_led_eidvstgrad.create(eid_pix_c, self.n_events_led, t_grad_c, n_tgrad, 'Event ID', 'Gradient Peak Time')
        self.p_led_tvstgrad.create(t_c, n_t, t_grad_c, n_tgrad, 'Peak Time', 'Gradient Peak Time')
        self.p_led_tvscharge.create(t_c, n_t, dl1_c, 100, 'Peak Time', 'Charge (p.e.)')
        self.p_led_tgradvscharge.create(t_grad_c, n_tgrad, dl1_c, 100, 'Gradient Peak Time', 'Charge (p.e.)')

        # 1D histograms wavg
        index = eid[eid == self.eoi].index[0]
        eoi_t = t_avg[index].compressed()
        eoi_tgrad = t_grad_avg[index].compressed()
        self.p_led_1deoicomp_wavg.create(eoi_tgrad, "Gradient Peak Time", "Peak Time Method Comparison (EventID = {})".format(self.eoi))
        self.p_led_1deoicomp_wavg.add(eoi_t, "Peak Time")
        t_shifted = (t_avg - t_avg.mean(1)[:, None]).compressed()
        t_grad_shifted = (t_grad_avg - t_grad_avg.mean(1)[:, None]).compressed()
        self.p_led_1dcomp_wavg.create(t_shifted, "Peak Time", "Peak Time Method Comparison (all events, shifted by mean of each event)")
        self.p_led_1dcomp_wavg.add(t_grad_shifted, "Gradient Peak Time")

        # 1D histograms
        index = eid[eid == self.eoi].index[0]
        eoi_t = t[index].compressed()
        eoi_tgrad = t_grad[index].compressed()
        self.p_led_1deoicomp.create(eoi_tgrad, "Gradient Peak Time", "Peak Time Method Comparison (EventID = {})".format(self.eoi))
        self.p_led_1deoicomp.add(eoi_t, "Peak Time")
        t_shifted = (t - t.mean(1)[:, None]).compressed()
        t_grad_shifted = (t_grad - t_grad.mean(1)[:, None]).compressed()
        self.p_led_1dcomp.create(t_shifted, "Peak Time", "Peak Time Method Comparison (all events, shifted by mean of each event)")
        self.p_led_1dcomp.add(t_grad_shifted, "Gradient Peak Time")

        # Camera Image
        eoi_tgrad = t_grad[index]
        self.p_led_imageeoitgrad.create(eoi_tgrad, "Gradient Peak Time", "Gradient Peak Time Across Camera")

        # LASER DATA
        df = self.df.loc[self.df['type'] == 'laser']

        eid = df['id']
        time = df['time']
        tack = df['tack']

        fci = np.ma.vstack(df['fci'])
        bp = np.ma.vstack(df['bp'])
        dl1 = np.ma.vstack(df['dl1'])
        t_avg = np.ma.vstack(df['t_avg'])
        t_grad_avg = np.ma.vstack(df['t_grad_avg'])
        t = np.ma.vstack(df['t'])
        t_grad = np.ma.vstack(df['t_grad'])
        fwhm = np.ma.vstack(df['fwhm'])
        mask = np.vstack(df['mask'])

        # bp.mask = mask
        dl1.mask = mask
        t_avg.mask = mask
        t_grad_avg.mask = mask
        t.mask = mask
        t_grad.mask = mask
        fwhm.mask = mask

        # 1D histograms wavg
        index = eid[eid == self.eoi].index[0]
        eoi_t = t_avg[index].compressed()
        eoi_tgrad = t_grad_avg[index].compressed()
        self.p_laser_1deoicomp_wavg.create(eoi_tgrad, "Gradient Peak Time", "Peak Time Method Comparison (EventID = {})".format(self.eoi))
        self.p_laser_1deoicomp_wavg.add(eoi_t, "Peak Time")
        t_shifted = (t_avg - t_avg.mean(1)[:, None]).compressed()
        t_grad_shifted = (t_grad_avg - t_grad_avg.mean(1)[:, None]).compressed()
        self.p_laser_1dcomp_wavg.create(t_shifted, "Peak Time", "Peak Time Method Comparison (all events, shifted by mean of each event)")
        self.p_laser_1dcomp_wavg.add(t_grad_shifted, "Gradient Peak Time")
        self.p_laser_1dcomp_wavg.ax.set_xlim([-5, 5])

        # 1D histograms
        index = eid[eid == self.eoi].index[0]
        eoi_t = t[index].compressed()
        eoi_tgrad = t_grad[index].compressed()
        self.p_laser_1deoicomp.create(eoi_tgrad, "Gradient Peak Time", "Peak Time Method Comparison (EventID = {})".format(self.eoi))
        self.p_laser_1deoicomp.add(eoi_t, "Peak Time")
        t_shifted = (t - t.mean(1)[:, None]).compressed()
        t_grad_shifted = (t_grad - t_grad.mean(1)[:, None]).compressed()
        self.p_laser_1dcomp.create(t_shifted, "Peak Time", "Peak Time Method Comparison (all events, shifted by mean of each event)")
        self.p_laser_1dcomp.add(t_grad_shifted, "Gradient Peak Time")

        # Camera Image
        eoi_tgrad = t_grad[index]
        self.p_laser_imageeoitgrad.create(eoi_tgrad, "Gradient Peak Time", "Gradient Peak Time Across Camera")

        # 1D histograms fwhm
        fwhm = fwhm.compressed()
        fwhm = fwhm[fwhm > 0]
        self.p_laser_fwhm.create(fwhm, "FWHM", "FWHM Distribution")

        # 1D histograms
        t_shifted = (t - t.mean(1)[:, None]).compressed()
        self.p_laser_1d_final.create(t_shifted, "Peak Time", "Smoothed Local Peak Time (all events)")

        # 1D histograms pixel
        t_shifted = (t - t.mean(1)[:, None])
        t_shifted_pix = t_shifted[:, 1825].compressed()
        self.p_laser_1d_final_pix.create(t_shifted_pix, "Pixel 1825", "Smoothed Local Peak Time (all events)", 10)
        t_shifted_pix = t_shifted[:, 1203].compressed()
        self.p_laser_1d_final_pix.add(t_shifted_pix, "Pixel 1203", 10)

        # self.p_led_eidvsfci.save()
        # self.p_led_timevstack.save()
        # self.p_led_bpvstack.save()
        # self.p_led_eidvst.save()
        # self.p_led_eidvstgrad.save()
        # self.p_led_tvstgrad.save()
        # self.p_led_tvscharge.save()
        # self.p_led_tgradvscharge.save()
        self.p_led_1deoicomp_wavg.save()
        self.p_led_1dcomp_wavg.save()
        self.p_led_1deoicomp.save()
        self.p_led_1dcomp.save()
        self.p_led_imageeoitgrad.save()

        self.p_laser_1deoicomp_wavg.save()
        self.p_laser_1dcomp_wavg.save()
        self.p_laser_1deoicomp.save()
        self.p_laser_1dcomp.save()
        self.p_laser_1d_final.save()
        self.p_laser_1d_final_pix.save()
        self.p_laser_imageeoitgrad.save()

        self.p_laser_fwhm.save()


exe = TimingExtractor()
exe.run()
