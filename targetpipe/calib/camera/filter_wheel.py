import numpy as np
from ctapipe.core import Component
from os.path import realpath, join, dirname
import pandas as pd


class FWCalibrator(Component):
    name = 'FWCalibrator'

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
        super().__init__(config=config, parent=tool, **kwargs)

        directory = join(realpath(dirname(__file__)), "../../io")
        fw_storage_path = join(directory, "FW.h5")
        store = pd.HDFStore(fw_storage_path)
        self.df = store['df']

    def get_illumination(self, fw):
        pe_log = np.log10(self.df['pe'])
        illumination = 10**np.interp(fw, self.df['position'], pe_log)
        return illumination
