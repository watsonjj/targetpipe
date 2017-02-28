import numpy as np
from traitlets import Unicode, observe
from ctapipe.core import Component
import target_calib


class TFApplier(Component):
    name = 'TFApplier'

    tf_path = Unicode(None, allow_none=True,
                      help='Path to the TargetCalib Transfer Function '
                           'file').tag(config=True)

    origin = None

    def __init__(self, config, tool, **kwargs):
        """
        Applier of the Transfer Function to waveforms

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
        if self.tf_path is None:
            raise ValueError("Please specify a path for the transfer "
                             "function file")

        self._load_tf()

    def _load_tf(self):
        self.calibrator = target_calib.Calibrator('', self.tf_path)

    @observe('tf_path')
    def on_tf_path_changed(self, change):
        new = change['new']
        try:
            self.log.warning("Change: tf_path={}".format(new))
            self._load_tf()
        except AttributeError:
            pass

    # def apply(self, waveforms, fci, tfwav):
    #     """
    #     Apply the transfer functions to the waveforms.
    #
    #     Parameters
    #     ----------
    #     waveforms : ndarray
    #         Numpy array containing the waveforms for an event.
    #         Size = (n_pix, n_samples) Type=np.uint16
    #     fci : ndarray
    #         Numpy array containing the first cell ids for each sample.
    #         Size = (n_pix) Type=np.uint16
    #     tfwav : ndarray
    #         Empty numpy array to be filled with the pedestal subtracted
    #         samples.
    #         Size = (n_pix) Type=np.uint16 or np.float32
    #
    #     """
    #
    #     pix2tm_m = self.pix2tm[self.mask]
    #     pix2tmpix_m = self.pix2tmpix[self.mask]
    #     waveforms_m = waveforms[self.mask]
    #     fci_m = fci[self.mask]
    #     pedsub_m = pedsub[self.mask]
    #     self.calibrator.ApplyEvent(pix2tm_m, pix2tmpix_m, waveforms_m, fci_m, pedsub_m)
    #     pedsub[self.mask] = pedsub_m

    def get_tf(self):
        """
        Obtain the pedestal value per pixel, per cell.

        Returns
        -------
        tf : ndarray
            Numpy array containing the pedestal values.
            Size = (n_pix, n_cells).

        """
        tf = np.array(self.calibrator.GetTf())
        adc_min = self.calibrator.GetAdcMin()
        adc_step = self.calibrator.GetAdcStep()
        return tf, adc_min, adc_step
