import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from traitlets import Dict, List
from scipy.stats import norm

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.image.charge_extractors import LocalPeakIntegrator
from ctapipe.core import Tool
from ctapipe.visualization import CameraDisplay

from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.calib.camera.r1 import TargetioR1Calibrator
from targetpipe.io.pixels import Dead, get_geometry
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
        self.fig.suptitle(title)
        self.ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))

        # axes[1].xaxis.set_minor_locator(AutoMinorLocator(5))
        # axes[2].yaxis.set_minor_locator(AutoMinorLocator(5))


class Hist2D(OfficialPlotter):
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


class WaveformHist1D(OfficialPlotter):
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

    def add(self, vals, label):
        mean, std = norm.fit(vals)
        fit_x = np.linspace(vals.min()-1, vals.max()+1, 1000)
        fit = norm.pdf(fit_x, mean, std)
        label = "{} (Mean = {:.3}, Stddev = {:.3})".format(label, mean, std)

        n = int((vals.max() - vals.min())*1.5+1)
        c = self.ax._get_lines.get_next_color()
        self.ax.hist(vals, n, normed=True, color=c, alpha=0.5)#, rwidth=0.5)
        self.ax.plot(fit_x, fit, color=c, alpha=0.8, label=label)

    def create(self, vals, label, title=""):

        self.add(vals, label)

        self.ax.set_xlabel("Time (ns)")
        self.ax.set_ylabel("Probability Density")
        self.ax.set_title(title)

        # self.ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        # self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    def save(self, output_path=None):
        self.ax.legend(loc=2)
        super().save(output_path)


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


class TimingExtractor(Tool):
    name = "TimingExtractor"
    description = "Loop through a file to extract the timing information"

    aliases = Dict(dict(max_events='TargetioFileReader.max_events',
                        ))
    classes = List([TargetioFileReader,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reader = None
        self.r1 = None
        self.dl0 = None
        self.dl1 = None
        self.dead = None

        self.df = None

        self.n_events = None
        self.n_pixels = None
        self.n_samples = None

        self.p_eidvsfci = None
        self.p_timevstack = None
        self.p_bpvstack = None
        self.p_eidvst = None
        self.p_eidvstgrad = None
        self.p_tvstgrad = None
        self.p_tvscharge = None
        self.p_tgradvscharge = None
        self.p_1deoicomp = None
        self.p_1dcomp = None
        self.p_imageeoitgrad = None

        self.eoi = 4

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        filepath = '/Volumes/gct-jason/data/170322/led/Run04345_r0.tio'
        self.reader = TargetioFileReader(input_path=filepath, **kwargs)

        extractor = LocalPeakIntegrator(**kwargs)

        self.r1 = TargetioR1Calibrator(pedestal_path='/Volumes/gct-jason/data/170322/pedestal/Run04240_ped.tcal',
                                       tf_path='/Volumes/gct-jason/data/170322/tf/Run04277-04327_tf.tcal',
                                       adc2pe_path='/Users/Jason/Software/CHECAnalysis/targetpipe/adc2pe/adc2pe_800gm_c1.tcal',
                                       **kwargs,
                                       )
        self.dl0 = CameraDL0Reducer(**kwargs)
        self.dl1 = CameraDL1Calibrator(extractor=extractor,
                                       **kwargs)

        self.dead = Dead()

        self.n_events = self.reader.num_events
        first_event = self.reader.get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        r1 = first_event.r1.tel[telid].pe_samples[0]
        self.n_pixels, self.n_samples = r1.shape

        p_kwargs = kwargs
        p_kwargs['script'] = "checm_paper_timing"
        p_kwargs['figure_name'] = "eid_vs_fci"
        self.p_eidvsfci = Scatter(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "time_vs_tack"
        self.p_timevstack = Scatter(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "bp_vs_tack"
        self.p_bpvstack = Hist2D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "eid_vs_t"
        self.p_eidvst = Hist2D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "eid_vs_tgrad"
        self.p_eidvstgrad = Hist2D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "t_vs_tgrad"
        self.p_tvstgrad = Hist2D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "t_vs_charge"
        self.p_tvscharge = Hist2D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "tgrad_vs_charge"
        self.p_tgradvscharge = Hist2D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "1D_comparison_eid{}".format(self.eoi)
        self.p_1deoicomp = WaveformHist1D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "1D_comparison_allevents".format(self.eoi)
        self.p_1dcomp = WaveformHist1D(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "image_tgrad_eid{}".format(self.eoi)
        self.p_imageeoitgrad = ImagePlotter(**p_kwargs)

    def start(self):
        dead = self.dead.get_pixel_mask()

        df_list = []

        run = self.reader.filename
        source = self.reader.read()
        desc = "Processing Events"
        for event in tqdm(source, total=self.n_events, desc=desc):
            ev = event.count
            event_id = event.r0.event_id
            time = event.trig.gps_time.value
            tack = event.meta['tack']
            fci = np.copy(event.r0.tel[0].first_cell_ids)
            bp = event.r0.tel[0].blockphase

            self.r1.calibrate(event)
            self.dl0.reduce(event)
            self.dl1.calibrate(event)
            r0 = event.r0.tel[0].adc_samples[0]
            r1 = event.r1.tel[0].pe_samples[0]
            dl1 = event.dl1.tel[0].image[0]
            grad = np.gradient(r1)[1]

            saturated = np.any(r0 < 10, 1)
            low_pe = np.all(r1 < 10, 1)
            mask = dead | saturated | low_pe

            ind = np.indices(r1.shape)[1]

            t_max = np.argmax(r1, 1)
            t_start = t_max - 2
            t_end = t_max + 2
            t_window = (ind >= t_start[..., None]) & (ind < t_end[..., None])
            t_windowed = np.ma.array(r1, mask=~t_window)
            t_windowed_ind = np.ma.array(ind, mask=~t_window)
            t = np.ma.average(t_windowed_ind, weights=t_windowed, axis=1)

            t_grad_max = np.argmax(grad, 1)
            t_grad_start = t_grad_max - 2
            t_grad_end = t_grad_max + 2
            t_grad_window = (ind >= t_grad_start[..., None]) & (ind < t_grad_end[..., None])
            t_grad_windowed = np.ma.array(grad, mask=~t_grad_window)
            t_grad_windowed_ind = np.ma.array(ind, mask=~t_grad_window)
            t_grad = np.ma.average(t_grad_windowed_ind, weights=t_grad_windowed, axis=1)

            # if (t_grad > 60).any():
            #     print(event_id)
            #
            if event_id == 23:
                continue

            d = dict(run=run,
                     index=ev,
                     id=event_id,
                     time=time,
                     tack=tack,
                     fci=fci,
                     bp=bp,
                     mask=mask,
                     dl1=dl1,
                     t=t,
                     t_grad=t_grad
                     )
            df_list.append(d)

        self.df = pd.DataFrame(df_list)
        store = pd.HDFStore('/Users/Jason/Downloads/timing.h5')
        store['df'] = self.df

        store = pd.HDFStore('/Users/Jason/Downloads/timing.h5')
        self.df = store['df']

    def finish(self):
        df = self.df

        eid = df['id']
        time = df['time']
        tack = df['tack']

        fci = np.ma.vstack(df['fci'])
        bp = np.ma.vstack(df['bp'])
        dl1 = np.ma.vstack(df['dl1'])
        t = np.ma.vstack(df['t'])
        t_grad = np.ma.vstack(df['t_grad'])
        mask = np.vstack(df['mask'])

        # bp.mask = mask
        dl1.mask = mask
        t.mask = mask
        t_grad.mask = mask

        # Scatter
        self.p_eidvsfci.create(eid, fci, 'Event ID', 'First Cell ID')
        self.p_timevstack.create(time, tack, 'Time', 'Tack')

        # 2D histograms
        eid_pix = eid[:, None] * np.ma.ones((eid.size, self.n_pixels))
        eid_pix.mask = mask
        eid_pix_c = eid_pix.compressed()
        tack_pix = tack[:, None] * np.ma.ones((tack.size, self.n_pixels))
        # tack_pix.mask = mask
        tack_pix_c = tack_pix.compressed()
        bp_c = bp.compressed()
        dl1_c = dl1.compressed()
        t_c = t.compressed()
        t_grad_c = t_grad.compressed()
        n_bp = int(bp_c.max() - bp_c.min())
        n_t = int(t_c.max() - t_c.min())
        n_tgrad = int(t_grad_c.max() - t_grad_c.min())
        self.p_bpvstack.create(bp_c, n_bp, tack_pix_c, self.n_events, 'Blockphase', 'Tack')
        self.p_eidvst.create(eid_pix_c, self.n_events, t_c, n_t, 'Event ID', 'Peak Time')
        self.p_eidvstgrad.create(eid_pix_c, self.n_events, t_grad_c, n_tgrad, 'Event ID', 'Gradient Peak Time')
        self.p_tvstgrad.create(t_c, n_t, t_grad_c, n_tgrad, 'Peak Time', 'Gradient Peak Time')
        self.p_tvscharge.create(t_c, n_t, dl1_c, 100, 'Peak Time', 'Charge (p.e.)')
        self.p_tgradvscharge.create(t_grad_c, n_tgrad, dl1_c, 100, 'Gradient Peak Time', 'Charge (p.e.)')

        # 1D histograms
        index = eid[eid == self.eoi].index[0]
        eoi_t = t[index].compressed()
        eoi_tgrad = t_grad[index].compressed()
        self.p_1deoicomp.create(eoi_tgrad, "Gradient Peak Time", "Peak Time Method Comparison (EventID = {})".format(self.eoi))
        self.p_1deoicomp.add(eoi_t, "Peak Time")
        t_shifted = (t - t.mean(1)[:, None]).compressed()
        t_grad_shifted = (t_grad - t_grad.mean(1)[:, None]).compressed()
        self.p_1dcomp.create(t_shifted, "Gradient Peak Time", "Peak Time Method Comparison (all events, shifted by mean of each event)")
        self.p_1dcomp.add(t_grad_shifted, "Peak Time")


        # Camera Image
        eoi_tgrad = t_grad[index]
        self.p_imageeoitgrad.create(eoi_tgrad, "Gradient Peak Time", "Gradient Peak Time Across Camera")

        # self.p_eidvsfci.save()
        # self.p_timevstack.save()
        # self.p_bpvstack.save()
        # self.p_eidvst.save()
        # self.p_eidvstgrad.save()
        # self.p_tvstgrad.save()
        # self.p_tvscharge.save()
        # self.p_tgradvscharge.save()
        self.p_1deoicomp.save()
        self.p_1dcomp.save()
        # self.p_imageeoitgrad.save()


exe = TimingExtractor()
exe.run()
