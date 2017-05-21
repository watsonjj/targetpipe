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

        self.gm800 = [819, 775, 784, 844, 772, 761, 785, 865, 863, 881, 715, 788, 725, 801, 828, 844, 793, 792, 835, 773, 774, 862, 803, 864, 788, 848, 766, 820, 773, 789, 845, 819]
        self.gm900 = [914, 864, 885, 950, 869, 851, 887, 974, 977, 991, 812, 885, 819, 897, 932, 946, 892, 886, 943, 871, 866, 968, 902, 974, 883, 959, 864, 925, 873, 891, 952, 925]
        self.gm1000 = [1009, 954, 985, 1057, 966, 940, 989, 1084, 1092, 1100, 910, 981, 913, 992, 1035, 1048, 991, 979, 1052, 969, 957, 1075, 1001, 1085, 977, 1071, 963, 1030, 973, 993, 1059, 1032]
        self.gm800_c1 = [850, 850, 850, 850, 850, 850, 850, 865, 863, 881, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 864, 850, 850, 850, 850, 850, 850, 850, 850]

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
