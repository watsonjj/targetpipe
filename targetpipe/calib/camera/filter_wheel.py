import numpy as np
from ctapipe.core import Component
from os.path import realpath, join, dirname, exists
import pandas as pd
from traitlets import Unicode


class FWCalibrator(Component):
    name = 'FWCalibrator'

    fw_path = Unicode(join(realpath(dirname(__file__)), "../../io", "FW.h5"),
                      allow_none=False,
                      help='Path to the FW h5 file').tag(config=True)

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

        if not exists(self.fw_path):
            self.log.error("No file FW file at: {}".format(self.fw_path))
            raise FileNotFoundError
        store = pd.HDFStore(self.fw_path)
        self.df = store['df']

    def load_from_txt(self, path):
        columns = ['position', 'attenuation_mean', 'attenuation_rms']
        self.df = pd.read_table(path, sep=' ', names=columns,
                                usecols=[0, 1, 2], skiprows=1)
        self.df = self.df.groupby('position').apply(np.mean)

    def set_calibration(self, fw, illumination):
        att_log = np.log10(self.df['attenuation_mean'])
        fw_att = 10**np.interp(fw, self.df['position'], att_log)
        i0 = illumination/(1 - fw_att)
        self.df = self.df.assign(pe=(1-self.df['attenuation_mean']) * i0)
        self.df = self.df.assign(pe_err=self.df['attenuation_rms'] * i0)

    def get_illumination(self, fw):
        try:
            pe_log = np.log10(self.df['pe'])
            illumination = 10 ** np.interp(fw, self.df['position'], pe_log)
            return illumination
        except KeyError:
            self.log.error("This FW file has no pe column, "
                           "need to use set_calibration()")

    def save(self, path):
        self.log.info("Storing fw calibration file: {}".format(path))
        store = pd.HDFStore(path)
        store['df'] = self.df
