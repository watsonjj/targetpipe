import numpy as np
from traitlets import Unicode, observe
from ctapipe.core import Component
import target_calib


class PedestalSubtractor(Component):
    name = 'PedestalSubtractor'

    pedestal_path = Unicode(None, allow_none=True,
                            help='Path to the TargetCalib pedestal '
                                 'file').tag(config=True)

    origin = None

    def __init__(self, config, tool, **kwargs):
        """
        Parent class for the r1 calibrators. Fills the r1 container.

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
        if self.pedestal_path is None:
            raise ValueError("Please specify a path for pedestal file")

        self._load_pedestal()

    def _load_pedestal(self):
        self.calibrator = target_calib.Calibrator(self.pedestal_path, '')

    @observe('pedestal_path')
    def on_input_path_changed(self, change):
        new = change['new']
        try:
            self.log.warning("Change: pedestal_path={}".format(new))
            self._load_pedestal()
        except AttributeError:
            pass

    def apply(self, event, pedsub):
        """
        Subtract the pedestal from the waveforms.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        pedsub : ndarray
            Empty numpy array to be filled with the pedestal subtracted
            samples.
            Size = (n_pix) Type=np.uint16 or np.float32

        """
        telid = 0
        tm = event.meta['tm']
        tmpix = event.meta['tmpix']
        waveforms = event.r0.tel[telid].adc_samples[0]
        fci = event.r0.tel[telid].first_cell_ids
        self.calibrator.ApplyEvent(tm, tmpix, waveforms, fci, pedsub)

    def get_ped(self):
        """
        Obtain the pedestal value per pixel, per cell.

        Returns
        -------
        pedestal : ndarray
            Numpy array containing the pedestal values.
            Size = (n_pix, n_cells).

        """
        return np.array(self.calibrator.GetPed())
