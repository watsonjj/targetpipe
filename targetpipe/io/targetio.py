"""Module to handle to storage of events extracted with `target_io` into
containers defined in `ctapipe.io.containers`.
"""

from target_io import TargetIOEventReader as TIOReader, \
    T_SAMPLES_PER_WAVEFORM_BLOCK as N_BLOCKSAMPLES
import numpy as np

from astropy import units as u
from astropy.time import Time

from targetpipe.io.containers import CHECDataContainer as DataContainer
from targetpipe.io.pixels import checm_pixel_pos, optical_foclen, \
    checm_refshape, checm_refstep, checm_time_slice

# CHEC-M
N_ROWS = 8
N_COLUMNS = 64
N_BLOCKS = N_ROWS * N_COLUMNS
N_CELLS = N_ROWS * N_COLUMNS * N_BLOCKSAMPLES
SKIP_SAMPLE = 32
SKIP_END_SAMPLE = 0
SKIP_EVENT = 2
SKIP_END_EVENT = 1

# OTHER
# N_ROWS = 8
# N_COLUMNS = 16
# N_BLOCKS = N_ROWS * N_COLUMNS
# N_CELLS = N_ROWS * N_COLUMNS * N_BLOCKSAMPLES
# SKIP_SAMPLE = 0
# SKIP_END_SAMPLE = 0
# SKIP_EVENT = 0
# SKIP_END_EVENT = 0


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

        self.tio_reader = TIOReader(self.url, N_CELLS,
                                    SKIP_SAMPLE, SKIP_END_SAMPLE,
                                    SKIP_EVENT, SKIP_END_EVENT)
        self.n_events = self.tio_reader.fNEvents
        self.run_id = self.tio_reader.fRunID
        self.n_pix = self.tio_reader.fNPixels
        self.n_modules = self.tio_reader.fNModules
        self.n_tmpix = self.n_pix // self.n_modules
        self.n_samples = self.tio_reader.fNSamples
        self.n_cells = self.tio_reader.fNCells

        # Setup camera geom
        self.pixel_pos = checm_pixel_pos[:, :self.n_pix]
        self.optical_foclen = optical_foclen

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
        data.meta['n_cells'] = self.n_cells
        data.meta['n_modules'] = self.n_modules
        data.meta['tm'] = np.arange(self.n_pix,
                                    dtype=np.uint16) // self.n_tmpix
        data.meta['tmpix'] = np.arange(self.n_pix,
                                       dtype=np.uint16) % self.n_tmpix

        data.inst.pixel_pos[chec_tel] = self.pixel_pos * u.m
        data.inst.optical_foclen[chec_tel] = self.optical_foclen * u.m
        data.inst.num_channels[chec_tel] = 1
        data.inst.num_pixels[chec_tel] = self.n_pix
        # data.inst.num_samples[chec_tel] = targetio_extractor.n_samples

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

        data.mc.tel[chec_tel].reference_pulse_shape = checm_refshape
        data.mc.tel[chec_tel].meta['refstep'] = checm_refstep
        data.mc.tel[chec_tel].time_slice = checm_time_slice

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
        if use_event_id:
            # Obtaining event id not implemented
            self.event_index = requested_event
        else:
            self.event_index = requested_event
