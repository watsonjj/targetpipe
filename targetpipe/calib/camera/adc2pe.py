import numpy as np
from ctapipe.core import Component
from traitlets import Unicode


class TargetioADC2PECalibrator(Component):
    name = 'TargetioADC2PECalibrator'
    origin = 'targetio'

    spe_path = Unicode('', help='Path to the numpy spe file').tag(config=True)
    gain_matching_path = Unicode('', help='Path to the numpy gain matching '
                                          'file').tag(config=True)

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

        spe = np.load(self.spe_path)
        spe = np.ma.masked_where(spe == 0, spe)

        self.adc2pe = 1/spe

        gain_matching = np.load(self.gain_matching_path)
        self.alpha = gain_matching['alpha_pix']
        self.c = gain_matching['C_pix']
        self.alpha = np.ma.masked_where(self.alpha == 0, self.alpha)
        self.c = np.ma.masked_where(self.c == 0, self.c)

    def get_adc2pe_at_hv(self, hv, pix):
        """
        
        Parameters
        ----------
        hv
        pix

        Returns
        -------
        adc2pe_at_hv : ndarray

        """
        adc2pe_at_1100 = self.adc2pe[pix]
        adc_at_1100 = self.c[pix] * np.power(1100, self.alpha[pix])
        adc_at_hv = self.c[pix] * np.power(hv, self.alpha[pix])
        adc2pe_at_hv = adc2pe_at_1100 * adc_at_1100 / adc_at_hv
        return adc2pe_at_hv

    def calibrate_pix(self, hv, pix, adc):
        return adc * self.get_adc2pe_at_hv(hv, pix)
