"""
Create a pedestal file from an event file using the target_calib Pedestal
class
"""

from traitlets import Dict, List
from ctapipe.core import Tool, Component
from ctapipe.io.eventfilereader import EventFileReaderFactory
from targetpipe.calib.camera.makers import PedestalMaker
from targetpipe.calib.camera.tf import TFApplier
from targetpipe.plots.official import ChecmPaperPlotter
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
# import seaborn as sns
from os.path import join, dirname
from IPython import embed
from targetpipe.io.targetio import get_bp_r_c
from matplotlib.colors import LogNorm


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

    def create(self, tf, adc_min, adc_step, tm, tmpix, cell):
        pix_tf = tf[tm, tmpix, cell]

        x = adc_min + np.arange(pix_tf.size) * adc_step
        y = pix_tf
        self.ax.plot(x, y, 'o')
        self.ax.plot(x, y)
        self.ax.set_title("TM: {}, TMPIX: {}, SAMP_CELL: {}".format(tm, tmpix, cell))
        self.ax.set_xlabel("ADC")
        self.ax.set_ylabel("Amplitude (mV)")

class TFApplicationPlotter(ChecmPaperPlotter):
    name = 'TFApplicationPlotter'

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

    def create(self, tf, adc_min, adc_step, tm, tmpix, cell):
        pix_tf = tf[tm, tmpix, cell]

        x = adc_min + np.arange(pix_tf.size) * adc_step
        y = pix_tf
        self.ax.plot(x, y)
        self.ax.set_title("TM: {}, TMPIX: {}, SAMP_CELL: {}".format(tm, tmpix, cell))
        self.ax.set_xlabel("ADC")
        self.ax.set_ylabel("Calibrated ADC")


class TFLow2DPlotter(ChecmPaperPlotter):
    name = 'TFLow2DPlotter'

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

    def create(self, tf, x, x_range, y_range, xlabel, ylabel):
        x = np.tile(x, (*tf.shape[:-1], 1))
        y = tf.ravel()

        _, _, _, h = self.ax.hist2d(x.ravel(), tf.ravel(), bins=[100, 100], range=[x_range, y_range], cmap='viridis', norm=LogNorm())

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        c = self.fig.colorbar(h)
        c.set_label("N")


class PedestalBuilder(Tool):
    name = "PedestalBuilder"
    description = "Create the TargetCalib Pedestal file from waveforms"

    aliases = Dict(dict(tf='TFApplier.tf_path',
                        ))
    classes = List([TFApplier
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tf = None

        self.p_tf = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.tf = TFApplier(**kwargs)

        script = "plot_tf"
        self.p_tf = TFPlotter(**kwargs, shape="wide")
        self.p_low2dpix = TFLow2DPlotter(**kwargs, shape="wide")
        self.p_low2dpixgrad = TFLow2DPlotter(**kwargs, shape="wide")
        self.p_low2dcell = TFLow2DPlotter(**kwargs, shape="wide")
        self.p_low2dcellgrad = TFLow2DPlotter(**kwargs, shape="wide")

    def start(self):

        # Get TF
        tf, adc_min, adc_step = self.tf.get_tf()
        tf = np.array(tf)

        # Get TF Applied
        x_app = np.linspace(-500, 3500, 1000, dtype=np.float32)
        tf_app = np.zeros([*tf.shape[:3], x_app.size], dtype=np.float32)
        for tm in range(tf.shape[0]):
            for tmpix in range(tf.shape[1]):
                for cell in range(tf.shape[2]):
                    bp, r, c = get_bp_r_c(cell)
                    y_i = tf_app[tm, tmpix, cell]
                    self.tf.calibrator.ApplyArray(x_app, y_i, tm, tmpix, 0, cell)

        tm = 0
        tmpix = 0
        cell = 2

        self.p_tf.create(tf, adc_min, adc_step, tm, tmpix, cell)

        tf_pixmean = np.mean(tf_app, axis=2)
        x_range = [0, 200]
        y_range = [-10, 130]
        x_label = "ADC"
        y_label = "Amplitude (mV)"
        self.p_low2dpix.create(tf_pixmean, x_app, x_range, y_range, x_label, y_label)

        grad = np.gradient(tf_pixmean, x_app, axis=-1)
        x_range = [0, 200]
        y_range = [0, 1.6]
        x_label = "ADC"
        y_label = "TF Slope"
        self.p_low2dpixgrad.create(grad, x_app, x_range, y_range, x_label, y_label)

        tf_pix = tf_app[tm, tmpix]
        x_range = [0, 200]
        y_range = [-10, 130]
        x_label = "ADC"
        y_label = "Amplitude (mV)"
        self.p_low2dcell.create(tf_pix, x_app, x_range, y_range, x_label, y_label)

        grad = np.gradient(tf_pix, x_app, axis=-1)
        x_range = [0, 200]
        y_range = [0, 1.6]
        x_label = "ADC"
        y_label = "TF Slope"
        self.p_low2dcellgrad.create(grad, x_app, x_range, y_range, x_label, y_label)

    def finish(self):
        output_dir = join(dirname(self.tf.tf_path), "plot_tf")

        self.p_tf.save(join(output_dir, "tf.pdf"))
        self.p_low2dpix.save(join(output_dir, "low2dpix.pdf"))
        self.p_low2dpixgrad.save(join(output_dir, "low2dpixgrad.pdf"))
        self.p_low2dcell.save(join(output_dir, "low2dcell.pdf"))
        self.p_low2dcellgrad.save(join(output_dir, "low2dcellgrad.pdf"))

exe = PedestalBuilder()
exe.run()
