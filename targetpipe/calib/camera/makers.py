import os
import numpy as np
from traitlets import Unicode, Int, Bool, List
from ctapipe.core import Component
from target_calib import PedestalMaker as TCPedestalMaker
from target_calib import TfMaker as TCTfMaker
from targetpipe.stats.rolling import PedestalMeanStdDev
from targetpipe.calib.camera.pedestal import PedestalSubtractor


class PedestalMaker(Component):
    name = 'PedestalMaker'

    output_path = Unicode(None, allow_none=True,
                          help='Path to save the TargetCalib pedestal '
                               'file').tag(config=True)
    n_tms = Int(32, help='Number of TARGET modules connected').tag(config=True)
    n_blocks = Int(512, help='Number of blocks').tag(config=True)
    n_samples = Int(96, help='Number of samples').tag(config=True)
    diagnosis = Bool(False, help='Run diagnosis while creating '
                                 'file?').tag(config=True)
    compress = Bool(False, help='Compress the pedestal file? (store in uint16 '
                                'instead of floats').tag(config=True)
    stddev = Bool(False, help='Create a numpy file containing the standard '
                              'deviation of the pedestal').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        Generator of Pedestal files.

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
        if self.output_path is None:
            raise ValueError("Please specify an output path to save "
                             "pedestal file")

        self.ped_obj = TCPedestalMaker(self.n_tms, self.n_blocks,
                                       self.n_samples, self.diagnosis)
        self.ped_stats = None
        # if self.stddev:
        #     self.ped_stats = PedestalMeanStdDev(self.n_tms * 64,
        #                                         self.n_cells)

    def add_event(self, event):
        """
        Add an event into the pedestal.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        """
        telid = 0
        waveforms = event.r0.tel[telid].adc_samples[0]
        first_cell_ids = event.r0.tel[telid].first_cell_ids
        self.ped_obj.AddEvent(waveforms, first_cell_ids)
        if self.ped_stats:
            self.ped_stats.send_waveform(waveforms, first_cell_ids)

    def save(self):
        """
        Save the pedestal file.
        """
        self.log.info("Saving pedestal to: {}".format(self.output_path))
        self.ped_obj.Save(self.output_path, self.compress)

        if self.ped_stats:
            stddev_path = os.path.splitext(self.output_path)[0] + '_stddev.npy'
            self.log.info("Saving pedestal stddev to: {}".format(stddev_path))
            np.save(stddev_path, self.ped_stats.stddev)


class TFMaker(Component):
    name = 'TFMaker'

    vped_list = List(Int, None, allow_none=True,
                     help='List of the vped value for each input '
                          'file').tag(config=True)
    pedestal_path = Unicode(None, allow_none=True,
                            help='Path to the pedestal file (TF requires the '
                                 'pedestal to be first subtracted before '
                                 'generating').tag(config=True)
    adc_step = Int(8, help='Step in ADC that the TF file will be stored '
                           'in').tag(config=True)
    output_path = Unicode(None, allow_none=True,
                          help='Path to save the TargetCalib pedestal '
                               'file').tag(config=True)
    number_tms = Int(32, help='Number of TARGET modules '
                              'connected').tag(config=True)
    vped_zero = Int(1050, help='VPed value for the pedestal').tag(config=True)
    compress = Bool(False, help='Compress the TF file?').tag(config=True)
    tf_input = Bool(False, help='Create a numpy file containing the input TF '
                                'array before the switch of '
                                'axis').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        Generator of Transfer Function files.

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
        if self.vped_list is None:
            raise ValueError("Please supply vped_list")
        if self.pedestal_path is None:
            raise ValueError("Please specify a pedestal path")
        if self.output_path is None:
            raise ValueError("Please specify an output path to save "
                             "TF file")

        self.ped = PedestalSubtractor(config=config, tool=tool,
                                      pedestal_path=self.pedestal_path)

        vpeds = np.array(self.vped_list, dtype=np.uint16)
        self.tf_obj = TCTfMaker(vpeds, self.number_tms, self.vped_zero)
        self.current_vped = None

    def add_event(self, event, vped):
        """
        Add an event into the transfer function.

        Parameters
        ----------
        event : container
            A `ctapipe` event container
        vped: int
            The vped of file from which the event comes from
        """
        if self.current_vped != vped:
            self.current_vped = vped
            self.tf_obj.SetVpedIndex(vped)
        telid = 0
        tm = event.meta['tm']
        tmpix = event.meta['tmpix']
        waveforms = event.r0.tel[telid].adc_samples[0]
        first_cell_ids = event.r0.tel[telid].first_cell_ids
        pedsub = np.zeros(waveforms.shape, dtype=np.float32)
        self.ped.apply(event, pedsub)
        self.tf_obj.AddEvent(pedsub, first_cell_ids)

    def save(self):
        """
        Save the pedestal file.
        """
        self.log.info("Saving transfer function to: {}"
                      .format(self.output_path))
        self.tf_obj.Save(self.output_path, self.adc_step, self.compress)

        if self.tf_input:
            self.save_tf_input()

    def save_tf_input(self):
        tf_input = np.array(self.tf_obj.GetTf())
        vped_vector = np.array(self.tf_obj.GetVpedVector())
        tfinput_path = os.path.splitext(self.output_path)[0] + '_input.npy'
        vped_path = os.path.splitext(self.output_path)[0] + '_vped.npy'
        self.log.info("Saving TF input array to: {}".format(tfinput_path))
        np.save(tfinput_path, tf_input)
        self.log.info("Saving Vped vector to: {}".format(vped_path))
        np.save(vped_path, vped_vector)
