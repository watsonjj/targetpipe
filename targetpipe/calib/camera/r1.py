from traitlets import Unicode, observe
import target_calib
from ctapipe.calib.camera.r1 import CameraR1Calibrator
import numpy as np


class TargetioR1Calibrator(CameraR1Calibrator):
    name = 'TargetioR1Calibrator'
    origin = 'targetio'

    pedestal_path = Unicode('', allow_none=True,
                            help='Path to the TargetCalib pedestal '
                                 'file').tag(config=True)
    tf_path = Unicode('', allow_none=True,
                      help='Path to the TargetCalib Transfer Function '
                           'file').tag(config=True)
    adc2pe_path = Unicode('', allow_none=True,
                          help='Path to the numpy adc2pe '
                               'file').tag(config=True)

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
        super().__init__(config=config, tool=tool, **kwargs)

        self.calibrator = None
        self.telid = 0

        self._load_calib()

    def calibrate(self, event):
        """
        Fake function to satisfy abstract parent, this is overloaded by
        either fake_calibrate or real_calibrate.
        """
        pass

    def _load_calib(self):
        if self.pedestal_path:
            self.calibrator = target_calib.Calibrator(self.pedestal_path,
                                                      self.tf_path,
                                                      [self.adc2pe_path])
            self.calibrate = self.real_calibrate
        else:
            self.log.warning("No pedestal path supplied, "
                             "r1 samples will equal r0 samples.")
            self.calibrate = self.fake_calibrate

    @observe('pedestal_path')
    def on_input_path_changed(self, change):
        new = change['new']
        try:
            self.log.warning("Change: pedestal_path={}".format(new))
            self._load_calib()
        except AttributeError:
            pass

    @observe('tf_path')
    def on_tf_path_changed(self, change):
        new = change['new']
        try:
            self.log.warning("Change: tf_path={}".format(new))
            self._load_calib()
        except AttributeError:
            pass

    def fake_calibrate(self, event):
        if event.meta['origin'] != self.origin:
            raise ValueError('Using TargetioR1Calibrator to calibrate a '
                             'non-targetio event.')

        if self.check_r0_exists(event, self.telid):
            samples = event.r0.tel[self.telid].adc_samples
            event.r1.tel[self.telid].pe_samples = samples

    def real_calibrate(self, event):
        if event.meta['origin'] != self.origin:
            raise ValueError('Using TargetioR1Calibrator to calibrate a '
                             'non-targetio event.')

        if self.check_r0_exists(event, self.telid):
            samples = event.r0.tel[self.telid].adc_samples[0]
            fci = event.r0.tel[self.telid].first_cell_ids
            r1 = event.r1.tel[self.telid].pe_samples[0]
            self.calibrator.ApplyEvent(samples, fci, r1)
