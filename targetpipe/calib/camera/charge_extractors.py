import numpy as np
from ctapipe.core import Component
from traitlets import Int


class CHECMExtractor(Component):
    name = 'CHECMExtractor'

    width = Int(8, help='Define the width of the integration '
                         'window').tag(config=True)
    shift = Int(3, help='Define the shift of the integration window from the '
                        'peakpos (peakpos - shift).').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        Use a method for filtering the signal

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

        self.iw_l = None
        self.iw_r = None

    def extract(self, samples, t0):
        n_pixels, n_samples = samples.shape

        self.iw_l = t0 - self.shift
        self.iw_r = self.iw_l + self.width
        int_window = np.s_[self.iw_l:self.iw_r]

        # Extract charge
        peak_area = np.sum(samples[:, int_window], axis=1)
        peak_height = samples[np.arange(n_pixels), t0]
        # peak_t = np.argmax(sb_sub_wf[:, int_window], axis=1)

        return peak_area, peak_height
