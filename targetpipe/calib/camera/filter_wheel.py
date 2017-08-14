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

    def set_calibration(self, fw, illumination):
        t_log = np.log10(self.df['transmission'])
        transmission = 10**np.interp(fw, self.df['position'], t_log)
        i0 = illumination / transmission
        self.df = self.df.assign(illumination=self.df['transmission'] * i0)
        self.df = self.df.assign(illumination_err=self.df['error'] * i0)

    def get_illumination(self, fw):
        try:
            pe_log = np.log10(self.df['illumination'])
            illumination = 10 ** np.interp(fw, self.df['position'], pe_log)
            return illumination
        except KeyError:
            self.log.error("This FW file has no illumination column, "
                           "need to use set_calibration()")

    def get_illumination_err(self, fw):
        try:
            err_log = np.log10(self.df['illumination_err'])
            err = 10 ** np.interp(fw, self.df['position'], err_log)
            return err
        except KeyError:
            self.log.error("This FW file has no illumination column, "
                           "need to use set_calibration()")

    def save(self, path):
        self.log.info("Storing fw calibration file: {}".format(path))
        store = pd.HDFStore(path)
        store['df'] = self.df
