"""Module to handle to storage of events extracted with `target_io` into
containers defined in `ctapipe.io.containers`.
"""

import numpy as np
from astropy import units as u
from astropy.time import Time
from target_io import TargetIOEventReader as TIOReader, \
    T_SAMPLES_PER_WAVEFORM_BLOCK as N_BLOCKSAMPLES

from ctapipe.instrument import TelescopeDescription
from targetpipe.io.camera import Config
from targetpipe.io.containers import CHECDataContainer as DataContainer


def get_bp_r_c(cells):
    blockphase = cells % N_BLOCKSAMPLES
    row = (cells // N_BLOCKSAMPLES) % 8
    column = (cells // N_BLOCKSAMPLES) // 8
    return blockphase, row, column


class TargetioExtractor:
    """
    Extract waveforms from `target_io` and build them into a camera image

    Attributes
    ----------
    tio_reader : target_io.TargetIOEventReader()
    n_events : int
        number of events in the fits file
    n_samples : int
        number of samples in the waveform
    r0_samples : ndarray
        two dimensional array to store the waveform for each pixel

    """
    def __init__(self, url, max_events=None):
        """
        Parameters
        ----------
        url : string
            path to the TARGET fits file
        """
        self._event_index = None

        self.url = url
        self.max_events = max_events

        self.event_id = 0
        self.time_tack = None
        self.time_sec = None
        self.time_ns = None

        self.cameraconfig = Config()

        self.tio_reader = TIOReader(self.url,
                                    self.cameraconfig.n_cells,
                                    self.cameraconfig.skip_sample,
                                    self.cameraconfig.skip_end_sample,
                                    self.cameraconfig.skip_event,
                                    self.cameraconfig.skip_end_event)
        self.n_events = self.tio_reader.fNEvents
        first_event_id = self.tio_reader.fFirstEventID
        last_event_id = self.tio_reader.fLastEventID
        self.event_id_list = np.arange(first_event_id, last_event_id)
        self.run_id = self.tio_reader.fRunID
        self.n_pix = self.tio_reader.fNPixels
        self.n_modules = self.tio_reader.fNModules
        self.n_tmpix = self.n_pix // self.n_modules
        self.n_samples = self.tio_reader.fNSamples
        self.n_cells = self.tio_reader.fNCells

        # Setup camera geom
        if self.n_pix == self.n_tmpix:
            self.cameraconfig.switch_to_single_module()
        self.pixel_pos = self.cameraconfig.pixel_pos
        self.optical_foclen = self.cameraconfig.optical_foclen

        self.n_rows = self.cameraconfig.n_rows
        self.n_columns = self.cameraconfig.n_columns
        self.n_blocks = self.cameraconfig.n_blocks
        self.refshape = self.cameraconfig.refshape
        self.refstep = self.cameraconfig.refstep
        self.time_slice = self.cameraconfig.time_slice

        # Init arrays
        self.r0_samples = None
        self.r1_samples = np.zeros((self.n_pix, self.n_samples),
                                   dtype=np.float32)[None, ...]
        self.first_cell_ids = np.zeros(self.n_pix, dtype=np.uint16)

        # Setup if file is already r1
        self.is_r1 = self.tio_reader.fR1
        if self.is_r1:
            self.get_event = self.tio_reader.GetR1Event
            self.samples = self.r1_samples[0]
        else:
            self.r0_samples = np.zeros((self.n_pix, self.n_samples),
                                       dtype=np.uint16)[None, ...]
            self.get_event = self.tio_reader.GetR0Event
            self.samples = self.r0_samples[0]

        self.data = None
        self.init_container()

    @property
    def event_index(self):
        return self._event_index

    @event_index.setter
    def event_index(self, val):
        self._event_index = val
        self.get_event(self.event_index, self.samples, self.first_cell_ids)
        self.event_id = self.tio_reader.fCurrentEventID
        self.time_tack = self.tio_reader.fCurrentTimeTack
        self.time_sec = self.tio_reader.fCurrentTimeSec
        self.time_ns = self.tio_reader.fCurrentTimeNs
        self.update_container()

    def move_to_next_event(self):
        for self.event_index in range(self.n_events):
            yield self.run_id, self.event_id

    def init_container(self):
        url = self.url
        max_events = self.max_events
        chec_tel = 0

        data = DataContainer()
        data.meta['origin'] = "targetio"

        # some targetio_event_source specific parameters
        data.meta['input'] = url
        data.meta['max_events'] = max_events
        data.meta['n_rows'] = self.n_rows
        data.meta['n_columns'] = self.n_columns
        data.meta['n_blocks'] = self.n_blocks
        data.meta['n_blockphases'] = N_BLOCKSAMPLES
        data.meta['n_cells'] = self.n_cells
        data.meta['n_modules'] = self.n_modules
        data.meta['tm'] = np.arange(self.n_pix,
                                    dtype=np.uint16) // self.n_tmpix
        data.meta['tmpix'] = np.arange(self.n_pix,
                                       dtype=np.uint16) % self.n_tmpix

        pix_pos = self.pixel_pos * u.m
        foclen = self.optical_foclen * u.m
        teldesc = TelescopeDescription.guess(*pix_pos, foclen)
        data.inst.subarray.tels[chec_tel] = teldesc

        self.data = data

    def update_container(self):
        data = self.data
        chec_tel = 0

        event_id = self.event_id
        run_id = self.run_id

        data.r0.run_id = run_id
        data.r0.event_id = event_id
        data.r0.tels_with_data = {chec_tel}
        data.r1.run_id = run_id
        data.r1.event_id = event_id
        data.r1.tels_with_data = {chec_tel}
        data.dl0.run_id = run_id
        data.dl0.event_id = event_id
        data.dl0.tels_with_data = {chec_tel}

        data.trig.tels_with_trigger = [chec_tel]

        data.meta['tack'] = self.time_tack
        data.meta['sec'] = self.time_sec
        data.meta['ns'] = self.time_ns
        data.trig.gps_time = Time(self.time_sec * u.s, self.time_ns * u.ns,
                                  format='unix', scale='utc', precision=9)

        data.count = self.event_index

        # this should be done in a nicer way to not re-allocate the
        # data each time (right now it's just deleted and garbage
        # collected)

        data.r0.tel.clear()
        data.r1.tel.clear()
        data.dl0.tel.clear()
        data.dl1.tel.clear()
        data.mc.tel.clear()

        # load the data per telescope/chan
        data.r0.tel[chec_tel].adc_samples = self.r0_samples
        data.r1.tel[chec_tel].pe_samples = self.r1_samples
        data.r0.tel[chec_tel].first_cell_ids = self.first_cell_ids
        bp, r, c = get_bp_r_c(self.first_cell_ids)
        data.r0.tel[chec_tel].blockphase = bp
        data.r0.tel[chec_tel].row = r
        data.r0.tel[chec_tel].column = c
        data.r0.tel[chec_tel].num_samples = self.n_samples

        data.mc.tel[chec_tel].reference_pulse_shape = self.refshape
        data.mc.tel[chec_tel].meta['refstep'] = self.refstep
        data.mc.tel[chec_tel].time_slice = self.time_slice

        data.meta['n_blocks'] = self.n_blocks

    def read_generator(self):
        data = self.data
        n_events = self.n_events
        if self.max_events and self.max_events < self.n_events:
            n_events = self.max_events
        for self.event_index in range(n_events):
            yield data

    def read_event(self, requested_event, use_event_id=False):
        """
        Obtain a particular event from the targetio file.

        Parameters
        ----------
        requested_event : int
        use_event_id : bool
            If True ,'requested_event' now seeks for a particular events id
            instead of index
        """
        index = requested_event
        if use_event_id:
            # Obtaining event id not implemented
            index = self.tio_reader.GetEventIndex(requested_event)
        n_events = self.n_events
        if self.max_events and self.max_events < self.n_events:
            n_events = self.max_events
        if (index >= n_events) | (index < 0):
            raise RuntimeError("Outside event range")
        self.event_index = index
