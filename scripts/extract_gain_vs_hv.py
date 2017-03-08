from traitlets import Dict, List, Int, Unicode
from ctapipe.core import Tool, Component
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from targetpipe.io.file_looper import TargetioFileLooper
from targetpipe.calib.camera.waveform_cleaning import CHECMWaveformCleaner
from targetpipe.calib.camera.charge_extractors import CHECMExtractor
from targetpipe.fitting.checm import CHECMFitter
import numpy as np
from tqdm import tqdm
from os import makedirs
from os.path import join, exists, dirname


class GainVsHVExtractor(Tool):
    name = "GainVsHVExtractor"
    description = "Extract gain vs hv"

    hv_list = List(Int, None, allow_none=True,
                   help='List of the hv value for each input '
                        'file').tag(config=True)
    output_path = Unicode(None, allow_none=True,
                          help='Path to save the numpy array containing the '
                               'gain').tag(config=True)
    adc2pe_path = Unicode('', allow_none=True,
                          help='Path to the numpy adc2pe '
                               'file').tag(config=True)

    aliases = Dict(dict(f='TargetioFileLooper.single_file',
                        N='TargetioFileLooper.max_files',
                        max_events='TargetioFileLooper.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        O='GainVsHVExtractor.output_path',
                        pe='GainVsHVExtractor.adc2pe_path'
                        ))

    classes = List([TargetioFileLooper,
                    CameraR1CalibratorFactory,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._event = None
        self._event_index = None
        self._event_id = None
        self._active_pixel = None

        self.w_event_index = None
        self.layout = None

        self.file_looper = None
        self.r1 = None
        self.dl0 = None

        self.n_events = None
        self.n_pixels = None
        self.n_samples = None
        self.n_modules = 32

        self.cleaner = None
        self.extractor = None
        self.fitter = None

        self.gain = None
        self.gain_error = None

        self.adc2pe = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.file_looper = TargetioFileLooper(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin='targetio', **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.cleaner = CHECMWaveformCleaner(**kwargs)
        self.extractor = CHECMExtractor(**kwargs)
        self.fitter = CHECMFitter(**kwargs, brightness='bright')

        self.n_events = self.file_looper.num_events
        first_event = self.file_looper.file_reader_list[0].get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        r0 = first_event.r0.tel[telid].adc_samples[0]
        self.n_pixels, self.n_samples = r0.shape

        self.hv_list = self.hv_list[:self.file_looper.num_readers]
        assert(len(self.file_looper.file_reader_list) == len(self.hv_list))

        if self.adc2pe_path:
            self.adc2pe = np.load(self.adc2pe_path)

    def start(self):
        n_hv = len(self.hv_list)
        # Prepare storage array
        self.gain = np.zeros((n_hv, self.n_pixels))
        self.gain_error = np.zeros((n_hv, self.n_pixels))
        area_list = []

        telid = 0
        desc = "Extracting area from events"
        with tqdm(total=self.n_events, desc=desc) as pbar:
            for fn, fr in enumerate(self.get_next_file()):
                source = fr.read()
                area = np.zeros((fr.num_events, self.n_pixels))
                for ev, event in enumerate(source):
                    pbar.update(1)
                    self.r1.calibrate(event)
                    self.dl0.reduce(event)

                    dl0 = event.dl0.tel[telid].pe_samples[0]

                    # Perform CHECM Waveform Cleaning
                    sb_sub_wf, t0 = self.cleaner.apply(dl0)

                    # Perform CHECM Charge Extraction
                    area[ev], _ = self.extractor.extract(sb_sub_wf, t0)
                area_list.append(area)

                if self.adc2pe is not None:
                    area *= self.adc2pe[None, :]

        desc = "Extracting gain of pixels"
        with tqdm(total=n_hv * self.n_pixels, desc=desc) as pbar:
            for fn in range(n_hv):
                for pix in range(self.n_pixels):
                    pbar.update(1)
                    if not self.fitter.apply(area_list[fn][:, pix]):
                        self.log.warning("FN {} Pixel {} could not be fitted"
                                         .format(fn, pix))
                        continue
                    self.gain[fn, pix] = self.fitter.gain
                    self.gain_error[fn, pix] = self.fitter.gain_error


    def finish(self):
        # Save figures
        output_dir = dirname(self.output_path)
        if not exists(output_dir):
            self.log.info("Creating directory: {}".format(output_dir))
            makedirs(output_dir)

        np.savez(self.output_path, gain=self.gain,
                 gain_error=self.gain_error, hv=self.hv_list)
        self.log.info("Numpy array saved to: {}".format(self.output_path))

    def get_next_file(self):
        for fr in self.file_looper.file_reader_list:
            yield fr


exe = GainVsHVExtractor()
exe.run()
