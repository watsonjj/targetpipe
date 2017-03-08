from traitlets import Dict, List
from ctapipe.core import Tool
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from targetpipe.calib.camera.waveform_cleaning import CHECMWaveformCleaner
from targetpipe.calib.camera.charge_extractors import CHECMExtractor
from targetpipe.fitting.checm import CHECMFitter
import numpy as np
from tqdm import tqdm
from os.path import join, exists
from os import makedirs


class BokehSPE(Tool):
    name = "BokehSPE"
    description = "Extract the conversion from adc to pe and save as a " \
                  "numpy array"

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        ))
    classes = List([EventFileReaderFactory,
                    CameraR1CalibratorFactory,
                    CHECMFitter
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reader = None
        self.r1 = None
        self.dl0 = None

        self.cleaner = None
        self.extractor = None
        self.fitter = None

        self.output_dir = None

        self.adc2pe = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.reader = reader_class(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin=self.reader.origin,
                                               **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.cleaner = CHECMWaveformCleaner(**kwargs)
        self.extractor = CHECMExtractor(**kwargs)
        self.fitter = CHECMFitter(**kwargs)

        self.output_dir = join(self.reader.output_directory, "extract_adc2pe")
        if not exists(self.output_dir):
            self.log.info("Creating directory: {}".format(self.output_dir))
            makedirs(self.output_dir)

    def start(self):
        n_events = self.reader.num_events
        first_event = self.reader.get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        n_pixels, n_samples = first_event.r0.tel[telid].adc_samples[0].shape

        # Prepare storage array
        area = np.zeros((n_events, n_pixels))
        self.adc2pe = np.zeros(n_pixels)

        source = self.reader.read()
        desc = "Looping through file"
        with tqdm(total=n_events, desc=desc) as pbar:
            for event in source:
                pbar.update(1)
                index = event.count

                self.r1.calibrate(event)
                self.dl0.reduce(event)

                dl0 = np.copy(event.dl0.tel[telid].pe_samples[0])

                # Perform CHECM Waveform Cleaning
                sb_sub_wf, t0 = self.cleaner.apply(dl0)

                # Perform CHECM Charge Extraction
                peak_area, peak_height = self.extractor.extract(sb_sub_wf, t0)

                area[index] = peak_area

        desc = "Extracting gain of pixels"
        with tqdm(total=n_pixels, desc=desc) as pbar:
            for pix in range(n_pixels):
                pbar.update(1)
                if not self.fitter.apply(area[:, pix]):
                    self.log.warning("Pixel {} couldn't be fit".format(pix))
                    continue
                self.adc2pe[pix] = 1/self.fitter.gain

    def finish(self):
        output_path = join(self.output_dir, "adc2pe.npy")
        np.save(output_path, self.adc2pe)
        self.log.info("adc2pe array saved: {}".format(output_path))


exe = BokehSPE()
exe.run()
