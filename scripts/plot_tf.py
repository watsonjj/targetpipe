"""
Create a pedestal file from an event file using the target_calib Pedestal
class
"""

from traitlets import Dict, List
from ctapipe.core import Tool, Component
from ctapipe.io.eventfilereader import EventFileReaderFactory
from targetpipe.calib.camera.makers import PedestalMaker
from targetpipe.calib.camera.tf import TFApplier
from targetpipe.plots.official import OfficialPlotter
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
# import seaborn as sns
from os.path import join, dirname


class TFPlotter(OfficialPlotter):
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
        self.ax.plot(x, y)
        self.ax.set_title("TM: {}, TMPIX: {}, SAMP_CELL: {}".format(tm, tmpix, cell))
        self.ax.set_xlabel("ADC")
        self.ax.set_ylabel("Calibrated ADC")


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

    def start(self):

        # Get TF
        tf, adc_min, adc_step = self.tf.get_tf()
        tf = np.array(tf)

        tm = 10
        tmpix = 10
        cell = 0

        self.p_tf.create(tf, adc_min, adc_step, tm, tmpix, cell)

    def finish(self):
        output_dir = join(dirname(self.tf.tf_path), "plot_tf")

        output_path = join(output_dir, "tf.pdf")
        self.p_tf.save(output_path)

exe = PedestalBuilder()
exe.run()
