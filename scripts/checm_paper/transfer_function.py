"""
Create a pedestal file from an event file using the target_calib Pedestal
class
"""
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, \
    FuncFormatter, AutoMinorLocator
from traitlets import Dict, List
from ctapipe.core import Tool, Component
from ctapipe.io.eventfilereader import EventFileReaderFactory
from targetpipe.calib.camera.makers import PedestalMaker
from targetpipe.calib.camera.r1 import TargetioR1Calibrator
from targetpipe.calib.camera.tf import TFApplier
from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.plots.official import ChecmPaperPlotter
from tqdm import tqdm, trange
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from os.path import join, dirname
from IPython import embed
import pandas as pd
from scipy.stats import norm
from targetpipe.utils.dactov import checm_dac_to_volts


class TFPlotter(ChecmPaperPlotter):
    name = 'TFPlotter'

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

    def create(self, tf, tf_single, tf_min, tf_max, adc_min, adc_step, title):
        x = adc_min + np.arange(tf_min.size) * adc_step
        self.ax.fill_between(x, tf_min, tf_max, color='black', label='All Cells')
        # self.ax.plot(x, tf_avg, color='grey', label='Average', lw=2)
        self.ax.plot(x, tf_single, color='grey', label='Single Cell', lw=2)
        # self.ax.set_title(title)
        self.ax.set_xlabel("ADC - Pedestal")
        self.ax.set_ylabel("Amplitude (V)")
        self.ax.legend(loc=2)


class BeforeAfterHist(ChecmPaperPlotter):
    name = 'Hist1D'

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
        self.ax.xaxis.label.set_color('b')
        self.ax_add = self.ax.twiny()
        self.ax_add.spines["bottom"].set_position(("axes", -0.2))
        self.make_patch_spines_invisible(self.ax_add)
        self.ax_add.spines["bottom"].set_visible(True)
        self.ax_add.xaxis.set_label_position('bottom')
        self.ax_add.xaxis.set_ticks_position('bottom')
        self.ax_add.xaxis.label.set_color('g')
        self.ax_add.xaxis.set_minor_locator(AutoMinorLocator(5))

        tkw = dict(size=4, width=1.5)
        self.ax.tick_params(axis='x', colors='b', **tkw)
        self.ax_add.tick_params(axis='x', colors='g', **tkw)
        self.ax.tick_params(axis='y', **tkw)

        self.before = None
        self.after = None

    @staticmethod
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    def add_before(self, hist, edges, mean, std):
        between = (edges[1:] + edges[:-1]) / 2
        label = "Before (Mean = {:.3f}, Stddev = {:.3f})".format(mean, std)
        self.ax.hist(between, bins=edges, weights=hist, alpha=0.5, color='b')
        self.ax.set_xlabel("Amplitude before TF (ADC)")
        self.before = patches.Patch(color='b', label=label)

    def add_after(self, hist, edges, mean, std):
        between = (edges[1:] + edges[:-1]) / 2
        label = "After (Mean = {:.3f}, Stddev = {:.3f})".format(mean, std)
        self.ax_add.hist(between, bins=edges, weights=hist, alpha=0.5, color='g')
        self.ax_add.set_xlabel("Amplitude after TF (V)")
        self.after = patches.Patch(color='g', label=label)

    def create(self, title=""):
        self.ax.set_ylabel("Probability Density")
        # self.ax.set_title(title)

        # self.ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        # self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    def save(self, output_path=None):
        self.ax.legend(loc=2, handles=[self.before, self.after])
        super().save(output_path)


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
        self.ax.yaxis.label.set_color('b')
        self.ax_add = self.ax.twinx()
        self.ax_add.yaxis.label.set_color('g')
        self.ax_add.yaxis.set_minor_locator(AutoMinorLocator(5))

        tkw = dict(size=4, width=1.5)
        self.ax.tick_params(axis='y', colors='b', **tkw)
        self.ax_add.tick_params(axis='y', colors='g', **tkw)
        self.ax.tick_params(axis='x', **tkw)

    def add_before(self, x, y, x_err=None, y_err=None):
        c = self.ax._get_lines.get_next_color()
        (_, caps, _) = self.ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', mew=0.5, color=c, alpha=0.8, markersize=3, capsize=3)

        for cap in caps:
            cap.set_markeredgewidth(1)

        self.ax.set_ylabel("Amplitude before TF (ADC)")

    def add_after(self, x, y, x_err=None, y_err=None):
        c = self.ax._get_lines.get_next_color()
        (_, caps, _) = self.ax_add.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', mew=0.5, color=c, alpha=0.8, markersize=3, capsize=3)

        for cap in caps:
            cap.set_markeredgewidth(1)

        self.ax_add.set_ylabel("Amplitude after TF (V)")

    def create(self, x_label="", y_label="", title=""):
        self.ax.set_xlabel(x_label)
        # self.ax.set_ylabel(y_label)
        # self.fig.suptitle(title)


class TFInvestigator(Tool):
    name = "TFInvestigator"
    description = "Produce plots associated with the " \
                  "transfer function calibration"

    aliases = Dict(dict())
    classes = List([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.df_file = None
        self.tf = None
        self.r1_p = None
        self.r1_t = None

        self.n_pixels = None
        self.n_samples = None

        self.poi = [1825, 1203, -1]
        self.voi = [800, 1120, 2000]

        self.p_tf = None
        self.p_range_sp = {}
        self.p_hist_vp = {}
        self.p_scatter = {}

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        ped_path = "/Volumes/gct-jason/data/170314/pedestal/Run00053_ped.tcal"
        tf_path = "/Volumes/gct-jason/data/170314/tf/Run00001-00050_tf.tcal"
        self.tf = TFApplier(**kwargs, tf_path=tf_path)
        self.r1_p = TargetioR1Calibrator(pedestal_path=ped_path,
                                         **kwargs,
                                         )
        self.r1_t = TargetioR1Calibrator(pedestal_path=ped_path,
                                         tf_path=tf_path,
                                         **kwargs,
                                         )

        dfl = []
        bp = '/Volumes/gct-jason/data/170314/tf/Run{:05}_r0.tio'
        it = range(50)
        for i in it:
            vped = 800 + i * 40
            dfl.append(dict(path=bp.format(i+1), vped=vped))

        for d in dfl:
            d['reader'] = TargetioFileReader(input_path=d['path'], **kwargs)
        self.df_file = pd.DataFrame(dfl)

        first_event = dfl[0]['reader'].get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        r1 = first_event.r1.tel[telid].pe_samples[0]
        self.n_pixels, self.n_samples = r1.shape

        p_kwargs = kwargs
        p_kwargs['script'] = "checm_paper_tf"
        p_kwargs['figure_name'] = "tfrange"
        self.p_tf = TFPlotter(**kwargs, shape="square")
        for pix in self.poi:
            p_kwargs['figure_name'] = "tfrange_p{}".format(pix)
            if pix == -1:
                continue
            self.p_range_sp[pix] = TFPlotter(**kwargs, shape="square")
        for vped in self.voi:
            for pix in self.poi:
                p_kwargs['figure_name'] = "tfhist_v{}_p{}".format(vped, pix)
                if pix == -1:
                    p_kwargs['figure_name'] = "tfhist_v{}_wc".format(vped)
                self.p_hist_vp[(vped, pix)] = BeforeAfterHist(**kwargs, shape="square")
        for pix in self.poi:
            p_kwargs['figure_name'] = "tfscatter_p{}".format(pix)
            if pix == -1:
                p_kwargs['figure_name'] = "tfscatter_wc"
            self.p_scatter[pix] = Scatter(**kwargs, shape="square")

    def start(self):
        # df_list = []
        #
        # desc1 = 'Looping through files'
        # n_rows = len(self.df_file.index)
        # t = tqdm(self.df_file.iterrows(), total=n_rows, desc=desc1)
        # for index, row in t:
        #     path = row['path']
        #     reader = row['reader']
        #     vped = row['vped']
        #
        #     source = reader.read()
        #     n_events = reader.num_events
        #
        #     wfs_p = np.zeros((n_events, self.n_pixels, self.n_samples))
        #     wfs_t = np.zeros((n_events, self.n_pixels, self.n_samples))
        #
        #     desc2 = "Calibrating WF"
        #     for event in tqdm(source, desc=desc2, total=n_events):
        #         ev = event.count
        #         self.r1_p.calibrate(event)
        #         wfs_p[ev] = event.r1.tel[0].pe_samples[0]
        #         self.r1_t.calibrate(event)
        #         wfs_t[ev] = checm_dac_to_mv(event.r1.tel[0].pe_samples[0])
        #
        #     desc3 = "Aggregate information per pixel"
        #     for pix in trange(self.n_pixels, desc=desc3):
        #         wf_p = wfs_p[:, pix]
        #         wf_t = wfs_t[:, pix]
        #         mean_p = np.mean(wf_p)
        #         mean_t = np.mean(wf_t)
        #         std_p = np.std(wf_p)
        #         std_t = np.std(wf_t)
        #         hist_p, edges_p = np.histogram(wf_p)
        #         hist_t, edges_t = np.histogram(wf_t)
        #
        #         df_list.append(dict(vped=vped,
        #                             pixel=pix,
        #                             mean_p=mean_p,
        #                             mean_t=mean_t,
        #                             std_p=std_p,
        #                             std_t=std_t,
        #                             hist_p=hist_p,
        #                             hist_t=hist_t,
        #                             edges_p=edges_p,
        #                             edges_t=edges_t
        #                             ))
        #
        #     # Whole Camera
        #     pix=-1
        #     mean_p = np.mean(wfs_p)
        #     mean_t = np.mean(wfs_t)
        #     std_p = np.std(wfs_p)
        #     std_t = np.std(wfs_t)
        #     hist_p, edges_p = np.histogram(wfs_p)
        #     hist_t, edges_t = np.histogram(wfs_t)
        #
        #     df_list.append(dict(vped=vped,
        #                         pixel=pix,
        #                         mean_p=mean_p,
        #                         mean_t=mean_t,
        #                         std_p=std_p,
        #                         std_t=std_t,
        #                         hist_p=hist_p,
        #                         hist_t=hist_t,
        #                         edges_p=edges_p,
        #                         edges_t=edges_t
        #                         ))
        #
        # df = pd.DataFrame(df_list)
        # store = pd.HDFStore('/Users/Jason/Downloads/tf.h5')
        # store['df'] = df

        store = pd.HDFStore('/Users/Jason/Downloads/tf.h5')
        df = store['df']

        # Get TF
        tf, adc_min, adc_step = self.tf.get_tf()
        tf = checm_dac_to_volts(np.array(tf))

        # Create Plots
        tf_min = np.min(tf, axis=(0,1,2))
        tf_max = np.max(tf, axis=(0,1,2))
        title = "Transfer Function Range, Whole Camera"
        self.p_tf.create(tf, tf[10,10,0], tf_min, tf_max, adc_min, adc_step, title)
        for pix, f in self.p_range_sp.items():
            if pix == -1:
                continue
            tm = pix // 64
            tmpix = pix % 64
            tf_pix = tf[tm, tmpix]
            tf_min = np.min(tf_pix, axis=0)
            tf_max = np.max(tf_pix, axis=0)
            title = "Transfer Function Range, Pixel = {}".format(pix)
            f.create(tf_pix, tf_pix[0], tf_min, tf_max, adc_min, adc_step, title)

        for (vped, pix), f in self.p_hist_vp.items():
            df_vp = df.loc[(df['vped'] == vped) & (df['pixel'] == pix)]
            mean_p = df_vp['mean_p'].values[0]
            mean_t = df_vp['mean_t'].values[0]
            std_p = df_vp['std_p'].values[0]
            std_t = df_vp['std_t'].values[0]
            hist_p = df_vp['hist_p'].values[0]
            hist_t = df_vp['hist_t'].values[0]
            edges_p = df_vp['edges_p'].values[0]
            edges_t = df_vp['edges_t'].values[0]
            title = "VPED = {}, Pixel = {}".format(vped, pix)
            if pix == -1:
                title = "VPED = {}, Whole Camera".format(vped)
            f.create(title)
            f.add_before(hist_p, edges_p, mean_p, std_p)
            f.add_after(hist_t, edges_t, mean_t, std_t)

        for pix, f in self.p_scatter.items():
            title = "ADC Distribtions, Pixel = {}".format(pix)
            if pix == -1:
                title = "ADC Distributions, Whole Camera"
            f.create("VPED", "", title)

            df_p = df.loc[df['pixel'] == pix]
            x = df_p['vped']
            y = df_p['mean_p']
            y_err = df_p['std_p']
            f.add_before(x, y, None, y_err)
            y = df_p['mean_t']
            y_err = df_p['std_t']
            f.add_after(x, y, None, y_err)






    def finish(self):
        self.p_tf.save()
        for pix, f in self.p_range_sp.items():
            f.save()
        for (vped, pix), f in self.p_hist_vp.items():
            f.save()
        for pix, f in self.p_scatter.items():
            f.save()

exe = TFInvestigator()
exe.run()
