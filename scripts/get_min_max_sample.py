from tqdm import tqdm
from traitlets import Dict, List
import numpy as np

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from ctapipe.image.waveform_cleaning import WaveformCleanerFactory
from ctapipe.io.eventfilereader import EventFileReaderFactory


class EventFileLooper(Tool):
    name = "EventFileLooper"
    description = "Loop through the file and apply calibration. Intended as " \
                  "a test that the routines work, and a benchmark of speed."

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        pe='CameraR1CalibratorFactory.adc2pe_path',
                        extractor='ChargeExtractorFactory.extractor',
                        extractor_t0='ChargeExtractorFactory.t0',
                        window_width='ChargeExtractorFactory.window_width',
                        window_shift='ChargeExtractorFactory.window_shift',
                        sig_amp_cut_HG='ChargeExtractorFactory.sig_amp_cut_HG',
                        sig_amp_cut_LG='ChargeExtractorFactory.sig_amp_cut_LG',
                        lwt='ChargeExtractorFactory.lwt',
                        clip_amplitude='CameraDL1Calibrator.clip_amplitude',
                        radius='CameraDL1Calibrator.radius',
                        cleaner='WaveformCleanerFactory.cleaner',
                        cleaner_t0='WaveformCleanerFactory.t0',
                        ))
    classes = List([EventFileReaderFactory,
                    ChargeExtractorFactory,
                    CameraR1CalibratorFactory,
                    CameraDL1Calibrator,
                    WaveformCleanerFactory
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_reader = None
        self.r1 = None
        self.dl0 = None
        self.dl1 = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.file_reader = reader_class(**kwargs)

        extractor_factory = ChargeExtractorFactory(**kwargs)
        extractor_class = extractor_factory.get_class()
        extractor = extractor_class(**kwargs)

        cleaner_factory = WaveformCleanerFactory(**kwargs)
        cleaner_class = cleaner_factory.get_class()
        cleaner = cleaner_class(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin=self.file_reader.origin,
                                               **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(extractor=extractor,
                                       cleaner=cleaner,
                                       **kwargs)

    def start(self):
        n_events = self.file_reader.num_events
        min_arr = np.ma.zeros(n_events)
        max_arr = np.ma.zeros(n_events)
        mean_arr = np.ma.zeros(n_events)
        median_arr = np.ma.zeros(n_events)
        source = self.file_reader.read()
        desc = "Looping through file"
        for event in tqdm(source, total=n_events, desc=desc):
            ev = event.count
            r0 = event.r0.tel[0].adc_samples[0]
            event.r0.tel[0].adc_samples[0] = np.full(r0.shape, 4098)
            self.r1.calibrate(event)
            self.dl0.reduce(event)

            for t in event.dl0.tels_with_data:
                samples = event.dl0.tel[t].pe_samples[0]
                min_arr[ev] = np.min(samples)
                max_arr[ev] = np.max(samples)
                mean_arr[ev] = np.mean(samples)
                median_arr[ev] = np.median(samples)

        print("Min = {}".format(min_arr.min()))
        print("Max = {}".format(max_arr.max()))
        print("(MaxofMin = {})".format(min_arr.max()))
        print("(MinofMax = {})".format(max_arr.min()))
        print("Mean = {}".format(mean_arr.mean()))
        print("Median = {}".format(np.median(median_arr)))

    def finish(self):
        pass


if __name__ == '__main__':
    exe = EventFileLooper()
    exe.run()
