import time
import copy
from tqdm import tqdm
import numpy as np
from ctapipe.core import Tool
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.instrument import CameraGeometry
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.charge_extractors import ChargeExtractorFactory
from ctapipe.image import tailcuts_clean, hillas_parameters
from ctapipe.image.hillas import HillasParameterizationError, hillas_parameters_2
from targetpipe.utils.plotting import intensity_to_hex


# GLOBALS

TELID = None
CHAN = None
N_EVENTS = None
N_PIXELS = None
N_SAMPLES = None

PIXEL_POS = None

LIVE_DATA = {'index': None,
             'r0': None,
             'r1': None,
             'dl0': None,
             'dl1': None}  # arrays of n_pix colours

FREEZE_DATA = {'index': None,
               'writing': False,
               'r0': None,
               'r1': None,
               'dl0': None,
               'dl1': None}

HILLAS = {'width': np.zeros(100),
          'length': np.zeros(100),
          'size': np.zeros(100),
          'phi': np.zeros(100),
          'miss': np.zeros(100),
          'r': np.zeros(100)}

HILLAS_EDGES = {'width': np.histogram(-1, bins=100, range=[0, 1])[1],
                'length': np.histogram(-1, bins=100, range=[0, 1])[1],
                'size': np.histogram(-1, bins=100, range=[0, 1000])[1],
                'phi': np.histogram(-1, bins=100, range=[0, 4])[1],
                'miss': np.histogram(-1, bins=100, range=[0, 1])[1],
                'r': np.histogram(-1, bins=100, range=[0, 1])[1]}

PARAMS = {'integration_window': [7, 3]}


class WaveformThread(Tool):
    name = "WaveformThread"
    description = "Thread to loop through events and perform calibration " \
                  "and hillas reduction"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.origin = 'hessio'
        # self.input_path = '/Users/Jason/Software/outputs/sim_telarray/meudon_gamma/simtel_runmeudon_gamma_30tel_30deg_19.gz'

        self.origin = 'targetio'
        self.input_path = '/Volumes/gct-jason/data/170328/Run04990_r0.tio'

        self.extractor = None
        self.r1 = None
        self.dl0 = None
        self.dl1 = None
        self.viewer = None
        self.geom = None

        self.arrays_setup = False

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=None, tool=None)

        ext = 'LocalPeakIntegrator'
        ext = 'SimpleIntegrator'
        start = '41'
        width = '4'
        extractor_factory = ChargeExtractorFactory(**kwargs, extractor=ext, start=start, width=width)
        extractor_class = extractor_factory.get_class()
        self.extractor = extractor_class(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin=self.origin, **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(extractor=self.extractor, **kwargs)

    def start(self):
        pass

    def finish(self):
        self.begin_watch()

    def begin_watch(self):
        # Watch folder for file
        while True:
            file_reader = self.get_file()

            # Setup arrays if first file
            if not self.arrays_setup:
                self.setup_arrays(file_reader)
                self.arrays_setup = True

            self.waveform_loop(file_reader)

    def get_file(self):
        kwargs = dict(config=None, tool=None)

        reader_factory = EventFileReaderFactory(input_path=self.input_path,
                                                **kwargs)
        reader_class = reader_factory.get_class()
        file_reader = reader_class(input_path=self.input_path, **kwargs)

        return file_reader

    def setup_arrays(self, file_reader):
        global TELID
        global CHAN
        global N_PIXELS
        global N_SAMPLES
        global PIXEL_POS

        first_event = file_reader.get_event(0)
        TELID = list(first_event.r0.tels_with_data)[0]
        CHAN = 0
        N_PIXELS, N_SAMPLES = first_event.r0.tel[TELID].adc_samples[CHAN].shape
        PIXEL_POS = first_event.inst.pixel_pos[TELID]
        pos = first_event.inst.pixel_pos[TELID]
        foclen = first_event.inst.optical_foclen[TELID]
        self.geom = CameraGeometry.guess(*pos, foclen)

    def waveform_loop(self, file_reader):
        global N_EVENTS
        global LIVE_DATA
        global FREEZE_DATA

        # Open file
        # N_EVENTS = file_reader.num_events

        with tqdm() as pbar:
            source = file_reader.read()
            for event in source:
                #  TODO: Remove telid loop once hillas tested
                for telid in event.r0.tels_with_data:
                    pbar.update(1)

                    self.r1.calibrate(event)
                    self.dl0.reduce(event)
                    self.dl1.calibrate(event)

                    index = event.count
                    image = event.dl1.tel[telid].image[CHAN]
                    mask = tailcuts_clean(self.geom, image, 1, 8, 5)
                    cleaned = np.ma.masked_array(image, ~mask)
                    pos = event.inst.pixel_pos[telid]
                    try:
                        hillas = hillas_parameters(*pos, cleaned)
                        # hillas = hillas_parameters_2(*pos, cleaned)
                    except HillasParameterizationError:
                        print('HillasParameterizationError')
                        continue

                    # print(hillas[0].length)

                    live_d = dict(index=index, image=intensity_to_hex(cleaned))
                    live_d = dict(index=index, image=intensity_to_hex(image))
                    LIVE_DATA = live_d

                    freeze_d = dict(index=index, event=copy.deepcopy(event))
                    FREEZE_DATA = freeze_d

                    width = hillas.width
                    length = hillas.length
                    size = hillas.size
                    phi = hillas.phi
                    miss = hillas.miss
                    r = hillas.r

                    HILLAS['width'] += \
                        np.histogram(width, bins=100, range=[0, 1])[0]
                    HILLAS['length'] += \
                        np.histogram(length, bins=100, range=[0, 1])[0]
                    HILLAS['size'] += \
                        np.histogram(size, bins=100, range=[0, 1000])[0]
                    HILLAS['phi'] += \
                        np.histogram(phi, bins=100, range=[0, 4])[0]
                    HILLAS['miss'] += \
                        np.histogram(miss, bins=100, range=[0, 1])[0]
                    HILLAS['r'] += \
                        np.histogram(r, bins=100, range=[0, 1])[0]

                    # time.sleep(0.01)


def start_tool():
    exe = WaveformThread()
    exe.run('')
