from traitlets import Dict, List
from ctapipe.core import Tool
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from targetpipe.calib.camera.waveform_cleaning import CHECMWaveformCleaner
from targetpipe.calib.camera.charge_extractors import CHECMExtractor
from targetpipe.fitting.checm import CHECMFitterSPE
from targetpipe.io.pixels import Dead
import numpy as np
from tqdm import tqdm
from os.path import join, exists
from os import makedirs


class DL1Extractor(Tool):
    name = "DL1Extractor"
    description = "Extract the dl1 information and store into a numpy file"

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        ))
    classes = List([EventFileReaderFactory,
                    CameraR1CalibratorFactory,
                    CHECMFitterSPE
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reader = None
        self.r1 = None
        self.dl0 = None

        self.cleaner = None
        self.extractor = None
        self.fitter = None
        self.dead = None

        self.output_dir = None

        self.adc2pe = None

        self.charge = None

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
        self.fitter = CHECMFitterSPE(**kwargs)
        self.dead = Dead()

        self.output_dir = join(self.reader.output_directory, "extract_adc2pe")
        if not exists(self.output_dir):
            self.log.info("Creating directory: {}".format(self.output_dir))
            makedirs(self.output_dir)

        n_events = self.reader.num_events
        first_event = self.reader.get_event(0)
        n_pixels, n_samples = first_event.r0.tel[0].adc_samples[0].shape

        self.charge = np.zeros((n_events, n_pixels))

    def start(self):
        n_events = self.reader.num_events
        first_event = self.reader.get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        n_pixels, n_samples = first_event.r0.tel[telid].adc_samples[0].shape

        # Prepare storage array
        area = np.zeros((n_events, n_pixels))
        ratio = np.ma.zeros(n_pixels)
        ratio.mask = np.zeros(ratio.shape, dtype=np.bool)
        ratio.fill_value = 0

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

                self.charge[index] = peak_area

    def finish(self):
        output_path = self.reader.input_path.replace("_r0.tio", "_dl1.npz")
        output_path = output_path.replace("_r1.tio", "_dl1.tio")
        np.savez(output_path,
                 charge=self.charge
                 )
        self.log.info("DL1 Numpy array saved to: {}".format(output_path))

exe = DL1Extractor()
exe.run()