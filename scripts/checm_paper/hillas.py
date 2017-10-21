from targetpipe.io.camera import Config
Config('checm')

import numpy as np
from matplotlib.ticker import AutoMinorLocator
from numpy.linalg.linalg import LinAlgError
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from traitlets import Dict, List
from os.path import join
from traitlets.config.loader import Config

from ctapipe.calib.camera import CameraCalibrator
from ctapipe.core import Tool
from ctapipe.image import tailcuts_clean
from ctapipe.image.hillas import HillasParameterizationError, \
    hillas_parameters
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe.io.eventfilereader import HessioFileReader

from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.plots.official import ChecmPaperPlotter
from astropy import units as u


class CustomCameraDisplay(CameraDisplay):

    def __init__(
            self,
            geometry,
            image=None,
            ax=None,
            title=None,
            norm="lin",
            cmap=None,
            allow_pick=False,
            autoupdate=True,
            autoscale=True,
            antialiased=True,
            ):
        self.ellipse = None
        self.ellipse_t = None
        super().__init__(geometry, image, ax, title, norm, cmap, allow_pick,
                         autoupdate, autoscale, antialiased)

    def overlay_moments_update(self, momparams, **kwargs):
        """helper to overlay ellipse from a `reco.MomentParameters` structure
        Updates existing ellipse if it already exists

        Parameters
        ----------
        momparams: `reco.MomentParameters`
            structuring containing Hillas-style parameterization
        kwargs: key=value
            any style keywords to pass to matplotlib (e.g. color='red'
            or linewidth=6)
        """

        # strip off any units
        cen_x = u.Quantity(momparams.cen_x).value
        cen_y = u.Quantity(momparams.cen_y).value
        length = u.Quantity(momparams.length).value
        width = u.Quantity(momparams.width).value
        text = "({:.02f},{:.02f})\n [w={:.03f},l={:.03f}]"\
            .format(momparams.cen_x, momparams.cen_y,
                    momparams.width, momparams.length)

        if not self.ellipse:
            self.ellipse = self.add_ellipse(centroid=(cen_x, cen_y),
                                            length=length*2,
                                            width=width*2,
                                            angle=momparams.psi.rad,
                                            **kwargs)
            # self.ellipse_t = self.axes.text(cen_x, cen_y, text,
            #                                 color=self.ellipse.get_edgecolor())
        else:
            self.ellipse.center = cen_x, cen_y
            self.ellipse.height = width*2
            self.ellipse.width = length*2
            self.ellipse.angle = momparams.psi.deg
            self.ellipse.update(kwargs)
            # self.ellipse_t.set_position((cen_x, cen_y))
            # self.ellipse_t.set_text(text)
            # self.ellipse_t.set_color(self.ellipse.get_edgecolor())


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

        self.cb = None

        self.fig = plt.figure(figsize=(13, 8))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.fig.subplots_adjust(right=0.85)

    def create(self, df, geom):
        super().save()

        camera = CustomCameraDisplay(geom, ax=self.ax,
                                     image=np.zeros(2048),
                                     cmap='viridis')
        camera.add_colorbar(pad=-0.2)
        camera.colorbar.set_label("Amplitude (p.e.)", fontsize=20)

        with PdfPages(self.output_path) as pdf:
            n_rows = len(df.index)
            desc = "Saving image pages"
            for index, row in tqdm(df.iterrows(), total=n_rows, desc=desc):
                event_id = row['id']
                tel = row['tel']
                image = row['image']
                tc = row['tc']
                hillas = row['hillas']

                cleaned_image = np.ma.masked_array(image, mask=~tc)
                max_ = cleaned_image.max()  # np.percentile(dl1, 99.9)
                min_ = np.percentile(image, 0.1)

                camera.image = image
                camera.set_limits_minmax(min_, max_)
                camera.highlight_pixels(tc, 'white')
                # camera.overlay_moments_update(hillas, color='red', linewidth=2)
                self.ax.set_title("Event: {}, Tel: {}".format(event_id, tel))
                self.ax.axis('off')
                camera.colorbar.ax.tick_params(labelsize=30)

                pdf.savefig(self.fig)

    def save(self, output_path=None):
        pass


class PeakTimePlotter(ChecmPaperPlotter):
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

        self.cb = None

        self.fig = plt.figure(figsize=(13, 8))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.fig.subplots_adjust(right=0.85)

    def create(self, df, geom):
        super().save()

        camera = CameraDisplay(geom, ax=self.ax,
                               image=np.ma.zeros(2048),
                               cmap='viridis')
        camera.add_colorbar(pad=-0.2)
        camera.colorbar.set_label("Peak Time (ns)", fontsize=20)

        with PdfPages(self.output_path) as pdf:
            n_rows = len(df.index)
            desc = "Saving image pages"
            for index, row in tqdm(df.iterrows(), total=n_rows, desc=desc):
                event_id = row['id']
                tel = row['tel']
                image = row['peak_time']
                tc = row['tc']
                hillas = row['hillas']

                cleaned_image = np.ma.masked_array(image, mask=~tc)
                cleaned_image.fill_value = 0
                max_ = np.percentile(cleaned_image.compressed(), 99)
                min_ = np.percentile(cleaned_image.compressed(), 1)

                camera.image = cleaned_image
                camera.set_limits_minmax(min_, max_)
                camera.highlight_pixels(np.arange(2048), 'black', 1, 0.2)
                # camera.overlay_moments_update(hillas, color='red')
                self.ax.set_title("Event: {}, Tel: {}".format(event_id, tel))
                self.ax.axis('off')
                camera.colorbar.ax.tick_params(labelsize=30)

                pdf.savefig(self.fig)

    def save(self):
        pass


class AllImagePlotter(ImagePlotter):
    name = 'AllImagePlotter'

    def create(self, df, geom):
        super().create(df, geom)


class ZeroWidthImagePlotter(ImagePlotter):
    name = 'AllImagePlotter'

    def create(self, df, geom):
        df_sub = df.loc[df['h_width'] < 0.001]
        super().create(df_sub, geom)


class MuonImagePlotter(ImagePlotter):
    name = 'AllImagePlotter'

    def create(self, df, geom):
        df_sub = df.loc[(df['h_length'] > 0.027) & (df['h_width'] < 0.007)]
        super().create(df_sub, geom)


class BrightImagePlotter(ImagePlotter):
    name = 'BrightImagePlotter'

    def create(self, df, geom):
        df_sub = df.loc[df['h_size'] > 10000]
        super().create(df_sub, geom)


class CountPlotter(ChecmPaperPlotter):
    name = 'CountPlotter'

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

    def create(self, df, geom):

        count = np.stack(df['tc']).sum(0)
        camera = CameraDisplay(geom, ax=self.ax,
                               image=count,
                               cmap='viridis')
        camera.add_colorbar()
        camera.colorbar.set_label("Count")

        self.ax.set_title("Pixel Hits after Tailcuts For Run")
        self.ax.axis('off')


class WholeDist(ChecmPaperPlotter):
    name = 'WholeDist'

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

        self.fig = plt.figure(figsize=(14, 10))
        self.ax1 = self.fig.add_subplot(2, 3, 1)
        self.ax2 = self.fig.add_subplot(2, 3, 2)
        self.ax3 = self.fig.add_subplot(2, 3, 3)
        self.ax4 = self.fig.add_subplot(2, 3, 4)
        self.ax5 = self.fig.add_subplot(2, 3, 5)
        self.ax6 = self.fig.add_subplot(2, 3, 6)

        self.ax1.set_title("Width")
        self.ax2.set_title("Length")
        self.ax3.set_title("Size")
        self.ax4.set_title("Phi")
        self.ax5.set_title("Miss")
        self.ax6.set_title("R")

        self.fig.subplots_adjust(wspace=0.3)

    def create(self, df):

        # self.ax1.hist(df.h_width)
        # self.ax2.hist(df.h_length)
        # self.ax3.hist(df.h_size)
        # self.ax4.hist(df.h_phi)
        # self.ax5.hist(df.h_miss)
        # self.ax6.hist(df.h_r)

        try:
            sns.distplot(df.h_width, ax=self.ax1)
            sns.distplot(df.h_length, ax=self.ax2)
            sns.distplot(df.h_size, ax=self.ax3)
            sns.distplot(df.h_phi, ax=self.ax4)
            sns.distplot(df.h_miss, ax=self.ax5)
            sns.distplot(df.h_r, ax=self.ax6)
        except ZeroDivisionError:
            pass
        except LinAlgError:
            pass

        self.ax1.set_xlabel("")
        self.ax2.set_xlabel("")
        self.ax3.set_xlabel("")
        self.ax4.set_xlabel("")
        self.ax5.set_xlabel("")
        self.ax6.set_xlabel("")

        self.ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax3.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax3.yaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax4.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax4.yaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax5.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax5.yaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax6.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax6.yaxis.set_minor_locator(AutoMinorLocator(5))


class WidthVsLength(ChecmPaperPlotter):
    name = 'WidthVsLength'

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
        x = 'h_width'
        y = 'h_length'

        df_data = df.loc[df['type'] == 'Data']
        df_mc = df.loc[df['type'] == 'MC']

        jp = sns.jointplot(x=x, y=y, data=df_data, stat_func=None)
        jp.fig.set_figwidth(self.fig.get_figwidth())
        jp.fig.set_figheight(self.fig.get_figheight())
        self.fig = jp.fig
        axes = self.fig.get_axes()
        self.ax = axes[0]

        jp.x = df_mc[x]
        jp.y = df_mc[y]
        jp.plot_joint(plt.scatter, marker='x', c='r',
                      s=30, lw=1, alpha=0.2, label="MC")

        self.ax.set_xlabel("Width (degrees)")
        self.ax.set_ylabel("Length (degrees)")
        self.fig.suptitle("Width Vs. Length")
        self.ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))

        axes[1].xaxis.set_minor_locator(AutoMinorLocator(5))
        axes[2].yaxis.set_minor_locator(AutoMinorLocator(5))

        df_8 = df_data.loc[df_data['id'] == 8]
        self.ax.plot(df_8[x], df_8[y], color="black",
                     marker='o', markersize=6,
                     label="Event 8: NSB? Grazing CR?")
        df_48 = df_data.loc[df_data['id'] == 48]
        self.ax.plot(df_48[x], df_48[y], color="green",
                     marker='o', markersize=6,
                     label="Event 48: Direct CR Entry&Exit")
        df_119 = df_data.loc[df_data['id'] == 119]
        self.ax.plot(df_119[x], df_119[y], color="red",
                     marker='o', markersize=6,
                     label="Event 119: Bright Shower")
        df_126 = df_data.loc[df_data['id'] == 126]
        self.ax.plot(df_126[x], df_126[y], color="purple",
                     marker='o', markersize=6,
                     label="Event 126: Direct CR")
        df_138 = df_data.loc[df_data['id'] == 138]
        self.ax.plot(df_138[x], df_138[y], color="yellow",
                     marker='o', markersize=6,
                     label="Event 138: Shower")

        self.ax.legend(loc="upper right", prop={'size': 10})


class SizeVsLength(ChecmPaperPlotter):
    name = 'WidthVsLength'

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
        x = 'h_size'
        y = 'h_length'

        df_data = df.loc[df['type'] == 'Data']
        df_mc = df.loc[df['type'] == 'MC']

        jp = sns.jointplot(x=x, y=y, data=df_data, stat_func=None)
        jp.fig.set_figwidth(self.fig.get_figwidth())
        jp.fig.set_figheight(self.fig.get_figheight())
        self.fig = jp.fig
        axes = self.fig.get_axes()
        self.ax = axes[0]

        jp.x = df_mc[x]
        jp.y = df_mc[y]
        jp.plot_joint(plt.scatter, marker='x', c='r',
                      s=30, lw=1, alpha=0.2, label="MC")

        self.ax.set_xlabel("Size (p.e.)")
        self.ax.set_ylabel("Length (degrees)")
        self.fig.suptitle("Size Vs. Length")
        self.ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))

        axes[1].xaxis.set_minor_locator(AutoMinorLocator(5))
        axes[2].yaxis.set_minor_locator(AutoMinorLocator(5))

        df_8 = df_data.loc[df_data['id'] == 8]
        self.ax.plot(df_8[x], df_8[y], color="black",
                     marker='o', markersize=6,
                     label="Event 8: NSB? Grazing CR?")
        df_48 = df_data.loc[df_data['id'] == 48]
        self.ax.plot(df_48[x], df_48[y], color="green",
                     marker='o', markersize=6,
                     label="Event 48: Direct CR Entry&Exit")
        df_119 = df_data.loc[df_data['id'] == 119]
        self.ax.plot(df_119[x], df_119[y], color="red",
                     marker='o', markersize=6,
                     label="Event 119: Bright Shower")
        df_126 = df_data.loc[df_data['id'] == 126]
        self.ax.plot(df_126[x], df_126[y], color="purple",
                     marker='o', markersize=6,
                     label="Event 126: Direct CR")
        df_138 = df_data.loc[df_data['id'] == 138]
        self.ax.plot(df_138[x], df_138[y], color="yellow",
                     marker='o', markersize=6,
                     label="Event 138: Shower")

        self.ax.legend(loc="upper right", prop={'size': 10})


class WidthDivLength(ChecmPaperPlotter):
    name = 'WidthDivLength'

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
        sns.distplot(df.h_width / df.h_length, ax=self.ax)
        self.ax.set_title("Width/Length")


class LengthDivSize(ChecmPaperPlotter):
    name = 'LengthDivSize'

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
        sns.distplot(df.h_length / df.h_size, ax=self.ax)
        self.ax.set_title("Length/Size")


class PairPlotter(ChecmPaperPlotter):
    name = 'PairPlotter'

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
        vars_ = ['time', 'duration', 'h_size', 'h_cen_x', 'h_cen_y', 'h_length', 'h_width',
                 'h_r', 'h_phi', 'h_psi', 'h_miss']
        pair = sns.pairplot(df, vars=vars_)
        pair.fig.set_figwidth(self.fig.get_figwidth() * 10)
        pair.fig.set_figheight(self.fig.get_figheight() * 10)
        self.fig = pair.fig
        axes = self.fig.get_axes()
        self.ax = axes[0]


class LengthPlotter(ChecmPaperPlotter):
    name = 'LengthPlotter'

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
        vals_data = df.loc[df['type'] == 'Data', 'h_length']
        vals_mc = df.loc[df['type'] == 'MC', 'h_length']

        hist_d, edges_d = np.histogram(vals_data, bins='fd')
        between_d = (edges_d[1:] + edges_d[:-1]) / 2
        widths_d = (edges_d[1:] - edges_d[:-1])
        norm_d = 1 / np.sum(hist_d * widths_d)
        hist_d = hist_d * norm_d
        error_d = np.sqrt(hist_d) * norm_d

        hist_mc, edges_mc = np.histogram(vals_mc, bins='fd')
        between_mc = (edges_mc[1:] + edges_mc[:-1]) / 2
        widths_mc = (edges_mc[1:] - edges_mc[:-1])
        norm_mc = 1 / np.sum(hist_mc * widths_mc)
        hist_mc = hist_mc * norm_mc
        error_mc = np.sqrt(hist_mc) * norm_mc

        sns.distplot(vals_data, ax=self.ax, label="Data", bins=edges_d,
                     kde=False, norm_hist=True, color='b')
        sns.distplot(vals_mc, ax=self.ax, label="MC", bins=edges_mc,
                     kde=False, norm_hist=True, color='r')

        # _, caps, _ = self.ax.errorbar(between_d, hist_d, yerr=error_d,
        #                  linestyle='', ms=7, mew=1, mfc='None', mec='b',
        #                  lw=1, ecolor='b', label="Data", capsize=1)
        # for cap in caps:
        #     cap.set_markeredgewidth(1)
        # _, caps, _ = self.ax.errorbar(between_mc, hist_mc, yerr=error_mc,
        #                  linestyle='', ms=7, mew=1, mfc='None', mec='g',
        #                  lw=1, ecolor='g', label="MC", capsize=1)
        # for cap in caps:
        #     cap.set_markeredgewidth(1)

        self.ax.errorbar(between_d, hist_d, yerr=error_d,
                         linestyle='', ms=7, mew=1, mfc='None', mec='b',
                         lw=1, ecolor='b', capsize=0)
        self.ax.errorbar(between_mc, hist_mc, yerr=error_mc,
                         linestyle='', ms=7, mew=1, mfc='None', mec='r',
                         lw=1, ecolor='r', capsize=0)

        self.ax.set_xlabel("Length (degrees)")
        self.ax.set_ylabel("Density")
        self.ax.legend(loc="upper right", prop={'size': 10})


class WidthPlotter(ChecmPaperPlotter):
    name = 'WidthPlotter'

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
        vals_data = df.loc[df['type'] == 'Data', 'h_width']
        vals_mc = df.loc[df['type'] == 'MC', 'h_width']

        hist_d, edges_d = np.histogram(vals_data, bins='fd')
        between_d = (edges_d[1:] + edges_d[:-1]) / 2
        widths_d = (edges_d[1:] - edges_d[:-1])
        norm_d = 1 / np.sum(hist_d * widths_d)
        hist_d = hist_d * norm_d
        error_d = np.sqrt(hist_d) * norm_d

        hist_mc, edges_mc = np.histogram(vals_mc, bins='fd')
        between_mc = (edges_mc[1:] + edges_mc[:-1]) / 2
        widths_mc = (edges_mc[1:] - edges_mc[:-1])
        norm_mc = 1 / np.sum(hist_mc * widths_mc)
        hist_mc = hist_mc * norm_mc
        error_mc = np.sqrt(hist_mc) * norm_mc

        sns.distplot(vals_data, ax=self.ax, label="Data", bins=edges_d,
                     kde=False, norm_hist=True, color='b')
        sns.distplot(vals_mc, ax=self.ax, label="MC", bins=edges_mc,
                     kde=False, norm_hist=True, color='r')

        # _, caps, _ = self.ax.errorbar(between_d, hist_d, yerr=error_d,
        #                  linestyle='', ms=7, mew=1, mfc='None', mec='b',
        #                  lw=1, ecolor='b', label="Data", capsize=1)
        # for cap in caps:
        #     cap.set_markeredgewidth(1)
        # _, caps, _ = self.ax.errorbar(between_mc, hist_mc, yerr=error_mc,
        #                  linestyle='', ms=7, mew=1, mfc='None', mec='g',
        #                  lw=1, ecolor='g', label="MC", capsize=1)
        # for cap in caps:
        #     cap.set_markeredgewidth(1)

        self.ax.errorbar(between_d, hist_d, yerr=error_d,
                         linestyle='', ms=7, mew=1, mfc='None', mec='b',
                         lw=1, ecolor='b', capsize=0)
        self.ax.errorbar(between_mc, hist_mc, yerr=error_mc,
                         linestyle='', ms=7, mew=1, mfc='None', mec='r',
                         lw=1, ecolor='r', capsize=0)

        self.ax.set_xlabel("Width (degrees)")
        self.ax.set_ylabel("Density")
        self.ax.legend(loc="upper right", prop={'size': 10})


class HillasBuilder(Tool):
    name = "HillasBuilder"
    description = "Loop through a file to extract the hillas parameters"

    aliases = Dict(dict(max_events='TargetioFileReader.max_events',
                        ))
    classes = List([TargetioFileReader,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reader_df = None
        self.df = None

        self.p_allimage = None
        self.p_alltimeimage = None
        self.p_allmcimage = None
        self.p_zwimage = None
        self.p_zwmcimage = None
        self.p_muonimage = None
        self.p_brightimage = None
        self.p_countimage = None
        self.p_whole_dist = None
        self.p_widthvslength = None
        self.p_sizevslength = None
        self.p_widthdivlength = None
        self.p_lengthdivsize = None
        self.p_pair = None
        self.p_mc_pair = None
        self.p_length = None
        self.p_width = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"

        data_config = self.config.copy()
        data_config['WaveformCleanerFactory'] = Config(cleaner='CHECMWaveformCleanerLocal')
        mc_config = self.config.copy()

        data_kwargs = dict(config=data_config, tool=self)
        mc_kwargs = dict(config=mc_config, tool=self)

        filepath = '/Volumes/gct-jason/data/170330/onsky-mrk501/Run05477_r1.tio'
        reader = TargetioFileReader(input_path=filepath, **data_kwargs)
        filepath = '/Users/Jason/Software/outputs/sim_telarray/meudon_cr/simtel_proton_nsb50_thrs30_1petal_rndm015_heide.gz'
        # filepath = '/Users/Jason/Software/outputs/sim_telarray/meudon_cr/simtel_proton_nsb50_thrs30.gz'
        reader_mc = HessioFileReader(input_path=filepath, **mc_kwargs)

        calibrator = CameraCalibrator(origin=reader.origin,
                                      **data_kwargs)
        calibrator_mc = CameraCalibrator(origin=reader_mc.origin,
                                         **mc_kwargs)

        first_event = reader.get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        pos = first_event.inst.pixel_pos[telid]
        foclen = first_event.inst.optical_foclen[telid]
        geom = CameraGeometry.guess(*pos, foclen)

        first_event = reader_mc.get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        pos_mc = first_event.inst.pixel_pos[telid]
        foclen = first_event.inst.optical_foclen[telid]
        geom_mc = CameraGeometry.guess(*pos_mc, foclen)

        d1 = dict(type='Data', reader=reader, calibrator=calibrator,
                  pos=pos, geom=geom, t1=20, t2=10)
        d2 = dict(type='MC', reader=reader_mc, calibrator=calibrator_mc,
                  pos=pos_mc, geom=geom_mc, t1=20, t2=10)
        self.reader_df = pd.DataFrame([d1, d2])

        p_kwargs = data_kwargs
        p_kwargs['script'] = "checm_paper_hillas"
        p_kwargs['figure_name'] = "all_images"
        self.p_allimage = AllImagePlotter(**p_kwargs)
        p_kwargs['figure_name'] = "all_peak_time_images"
        self.p_alltimeimage = PeakTimePlotter(**p_kwargs)
        p_kwargs['figure_name'] = "all_mc_images"
        self.p_allmcimage = AllImagePlotter(**p_kwargs)
        p_kwargs['figure_name'] = "zero_width_images"
        self.p_zwimage = ZeroWidthImagePlotter(**p_kwargs)
        p_kwargs['figure_name'] = "zero_width_mc_images"
        self.p_zwmcimage = ZeroWidthImagePlotter(**p_kwargs)
        p_kwargs['figure_name'] = "muon_images"
        self.p_muonimage = MuonImagePlotter(**p_kwargs)
        p_kwargs['figure_name'] = "bright_images"
        self.p_brightimage = BrightImagePlotter(**p_kwargs)
        p_kwargs['figure_name'] = "count_image"
        self.p_countimage = CountPlotter(**p_kwargs)
        p_kwargs['figure_name'] = "whole_distribution"
        self.p_whole_dist = WholeDist(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "width_vs_length"
        self.p_widthvslength = WidthVsLength(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "size_vs_length"
        self.p_sizevslength = SizeVsLength(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "width_div_length"
        self.p_widthdivlength = WidthDivLength(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "length_div_size"
        self.p_lengthdivsize = LengthDivSize(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "pair_plot"
        self.p_pair = PairPlotter(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "pair_mc_plot"
        self.p_mc_pair = PairPlotter(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "length"
        self.p_length = LengthPlotter(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "width"
        self.p_width = WidthPlotter(**p_kwargs, shape='wide')

    def start(self):
        df_list = []

        it = self.reader_df.iterrows()
        n_rows = len(self.reader_df.index)
        desc = "Looping over files"
        for index, row in tqdm(it, total=n_rows, desc=desc):
            type_ = row['type']
            reader = row['reader']
            calibrator = row['calibrator']
            pos = row['pos']
            geom = row['geom']
            t1 = row['t1']
            t2 = row['t2']

            desc = "Processing Events"
            source = reader.read()
            n_events = reader.num_events
            for event in tqdm(source, total=n_events, desc=desc):
                for telid in event.r0.tels_with_data:
                    ev = event.count
                    event_id = event.r0.event_id
                    time = event.trig.gps_time.value
                    calibrator.calibrate(event)
                    image = event.dl1.tel[telid].image[0]

                    # Cleaning
                    tc = tailcuts_clean(geom, image, t1, t2)
                    if not tc.any():
                        # self.log.warning('No image')
                        continue
                    cleaned_dl1 = np.ma.masked_array(image, mask=~tc)

                    wf = event.dl1.tel[telid].cleaned[0]
                    peak_time = np.ma.argmax(wf, axis=1)
                    peak_time_m = np.ma.masked_array(peak_time, mask=~tc)
                    shower_duration = peak_time_m.max() - peak_time_m.min()

                    try:
                        hillas = hillas_parameters(*pos, cleaned_dl1)
                    except HillasParameterizationError:
                        # self.log.warning('HillasParameterizationError')
                        continue

                    if np.isnan(hillas.width):
                        # self.log.warning("Hillas width == NaN")
                        continue

                    d = dict(type=type_,
                             index=ev, id=event_id, time=time, tel=telid,
                             image=image, tc=tc, peak_time=peak_time,
                             duration=shower_duration,
                             hillas=hillas,
                             h_size=hillas.size,
                             h_cen_x=hillas.cen_x.value,
                             h_cen_y=hillas.cen_y.value,
                             h_length=hillas.length.value,
                             h_width=hillas.width.value,
                             h_r=hillas.r.value,
                             h_phi=hillas.phi.value,
                             h_psi=hillas.psi.value,
                             h_miss=hillas.miss.value,
                             h_skewness=hillas.skewness,
                             h_kurtosis=hillas.kurtosis
                             )
                    df_list.append(d)

        self.df = pd.DataFrame(df_list)
        store = pd.HDFStore('/Users/Jason/Downloads/hillas.h5')
        store['df'] = self.df

        store = pd.HDFStore('/Users/Jason/Downloads/hillas.h5')
        self.df = store['df']
        self.df.loc[:, 'h_width'] /= 40.344e-3
        self.df.loc[:, 'h_length'] /= 40.344e-3

    def finish(self):
        df_data = self.df.loc[self.df['type'] == 'Data']
        df_mc = self.df.loc[self.df['type'] == 'MC']
        rdf = self.reader_df
        geom_data = rdf.loc[rdf['type'] == "Data", "geom"].iloc[0]
        geom_mc = rdf.loc[rdf['type'] == "MC", "geom"].iloc[0]

        self.p_allimage.create(df_data, geom_data)
        self.p_alltimeimage.create(df_data, geom_data)
        # self.p_allmcimage.create(df_mc, geom_mc)
        # self.p_zwimage.create(df_data, geom_data)
        # self.p_zwmcimage.create(df_mc, geom_mc)
        # self.p_muonimage.create(df_data, geom_data)
        # self.p_brightimage.create(df_data, geom_data)
        # self.p_countimage.create(df_data, geom_data)
        # self.p_whole_dist.create(df_data)
        # self.p_widthvslength.create(self.df)
        # self.p_sizevslength.create(self.df)
        # self.p_widthdivlength.create(df_data)
        # self.p_lengthdivsize.create(df_data)
        # self.p_pair.create(df_data)
        # self.p_mc_pair.create(df_mc)
        # self.p_length.create(self.df)
        # self.p_width.create(self.df)

        self.p_allimage.save()
        self.p_alltimeimage.save()
        # self.p_allmcimage.save()
        # self.p_zwimage.save()
        # self.p_zwmcimage.save()
        # self.p_muonimage.save()
        # self.p_brightimage.save()
        # self.p_countimage.save()
        # self.p_whole_dist.save()
        # self.p_widthvslength.save()
        # self.p_sizevslength.save()
        # self.p_widthdivlength.save()
        # self.p_lengthdivsize.save()
        # self.p_pair.save()
        # self.p_mc_pair.save()
        # self.p_length.save()
        # self.p_width.save()

        output_dir = self.p_allimage.output_dir

        np_path = join(output_dir, "hillas.npz")
        np.savez("/Users/Jason/Downloads/hillas.npz",
                 time=df_data['time'],
                 size=df_data['h_size'],
                 cen_x=df_data['h_cen_x'],
                 cen_y=df_data['h_cen_y'],
                 length=df_data['h_length'],
                 width=df_data['h_width'],
                 r=df_data['h_r'],
                 phi=df_data['h_phi'],
                 psi=df_data['h_psi'],
                 miss=df_data['h_miss'],
                 skewness=df_data['h_skewness'],
                 kurtosis=df_data['h_kurtosis'])
        self.log.info("Created numpy file: {}".format(np_path))

        txt_path = join(output_dir, "hillas.csv")
        with open(txt_path, 'w') as f:
            f.write("time,size,cen_x,cen_y,length,width,r,"
                    "phi,psi,miss,skewness,kurtosis\n")
            for index, row in df_data.iterrows():
                f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n"
                        .format(row['time'],
                                row['h_size'],
                                row['h_cen_x'],
                                row['h_cen_y'],
                                row['h_length'],
                                row['h_width'],
                                row['h_r'],
                                row['h_phi'],
                                row['h_psi'],
                                row['h_miss'],
                                row['h_skewness'],
                                row['h_kurtosis']))
        self.log.info("Created txt file: {}".format(txt_path))


exe = HillasBuilder()
exe.run()
