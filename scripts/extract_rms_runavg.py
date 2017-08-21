from traitlets import Dict, List, Unicode
from ctapipe.core import Tool
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
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
                        pe='CameraR1CalibratorFactory.pe_path',
                        ))
    classes = List([EventFileReaderFactory,
                    CameraR1CalibratorFactory,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reader = None
        self.r1 = None
        self.dl0 = None

        self.output_dir = None

        self.baseline_rms_full = None

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

        self.output_dir = join(self.reader.output_directory, "extract_adc2pe")
        if not exists(self.output_dir):
            self.log.info("Creating directory: {}".format(self.output_dir))
            makedirs(self.output_dir)

        n_events = self.reader.num_events
        first_event = self.reader.get_event(0)
        n_pixels = first_event.inst.num_pixels[0]
        n_samples = first_event.r0.tel[0].num_samples

        self.baseline_rms_full = np.zeros((n_events, n_pixels))

    def start(self):
        n_events = self.reader.num_events
        first_event = self.reader.get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        n_pixels = first_event.inst.num_pixels[0]
        n_samples = first_event.r0.tel[0].num_samples

        source = self.reader.read()
        desc = "Looping through file"
        with tqdm(total=n_events, desc=desc) as pbar:
            for event in source:
                pbar.update(1)
                ev = event.count

                self.r1.calibrate(event)
                self.dl0.reduce(event)

                dl0 = event.dl0.tel[telid].pe_samples[0]

                baseline_rms_full = np.std(dl0, axis=1)

                self.baseline_rms_full[ev] = baseline_rms_full

    def finish(self):
        output_path = self.reader.input_path.replace("_r0.tio", "_rms_runavg.npz")
        output_path = output_path.replace("_r1.tio", "_rms_runavg.npy")
        np.save(output_path, np.mean(self.baseline_rms_full, axis=0))
        self.log.info("RMS Numpy array saved to: {}".format(output_path))

exe = DL1Extractor()
exe.run()
