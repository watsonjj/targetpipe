from os import makedirs
from os.path import exists, dirname

import numpy as np
from tqdm import tqdm
from tqdm import trange
from traitlets import Dict, List, Int, Unicode, Bool

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import AverageWfPeakIntegrator
from ctapipe.image.waveform_cleaning import CHECMWaveformCleanerAverage
from targetpipe.fitting.chec import CHECBrightFitter
from targetpipe.io.file_looper import TargetioFileLooper
from targetpipe.io.pixels import Dead


class ChargeVsRunExtractor(Tool):
    name = "ChargeVsRunExtractor"
    description = "Extract charge (gaussing fit mean of each pixel) " \
                  "vs some run descriptor."

    rundesc_list = List(Int, None, allow_none=True,
                        help='List of the description value for each input '
                             'file').tag(config=True)
    output_path = Unicode(None, allow_none=True,
                          help='Path to save the numpy array').tag(config=True)
    calc_mean = Bool(False, help='Extract the mean and stdev directly insted '
                                 'of fitting the file.').tag(config=True)

    aliases = Dict(dict(f='TargetioFileLooper.single_file',
                        N='TargetioFileLooper.max_files',
                        max_events='TargetioFileLooper.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        pe='CameraR1CalibratorFactory.adc2pe_path',
                        O='ChargeVsRunExtractor.output_path',
                        ))
    flags = Dict(dict(mean=({'ChargeVsRunExtractor': {'calc_mean': True}},
                            'Extract the mean and stdev directly insted '
                            'of fitting the file.')
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
        self.cleaner = None
        self.extractor = None
        self.dl1 = None

        self.n_pixels = None
        self.n_samples = None
        self.n_modules = 32

        self.cleaner = None
        self.extractor = None
        self.fitter = None
        self.dead = None

        self.charge = None
        self.charge_err = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.file_looper = TargetioFileLooper(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin='targetio', **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)
        self.cleaner = CHECMWaveformCleanerAverage(**kwargs)
        self.extractor = AverageWfPeakIntegrator(**kwargs)
        self.dl0 = CameraDL0Reducer(**kwargs)
        self.fitter = CHECBrightFitter(**kwargs)
        self.dead = Dead()

        file_reader_list = self.file_looper.file_reader_list
        first_event = file_reader_list[0].get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        r0 = first_event.r0.tel[telid].adc_samples[0]
        self.n_pixels, self.n_samples = r0.shape

        self.rundesc_list = self.rundesc_list[:self.file_looper.num_readers]
        assert (len(file_reader_list) == len(self.rundesc_list))

    def start(self):
        n_rundesc = len(self.rundesc_list)
        # Prepare storage array
        self.charge = np.ma.zeros((n_rundesc, self.n_pixels))
        self.charge.mask = np.zeros(self.charge.shape, dtype=np.bool)
        self.charge.fill_value = 0
        self.charge_err = np.ma.zeros((n_rundesc, self.n_pixels))
        self.charge_err.mask = np.zeros(self.charge_err.shape, dtype=np.bool)
        self.charge_err.fill_value = 0

        telid = 0
        desc1 = "Looping over runs"
        iterable = enumerate(self.get_next_file())
        for fn, fr in tqdm(iterable, total=n_rundesc, desc=desc1):
            source = fr.read()
            n_events = fr.num_events
            area = np.zeros((n_events, self.n_pixels))
            desc2 = "Extracting area from events"
            for event in tqdm(source, total=n_events, desc=desc2):
                ev = event.count
                self.r1.calibrate(event)
                self.dl0.reduce(event)
                kwargs = dict(config=self.config, tool=self,
                              extractor=self.extractor, cleaner=self.cleaner)
                dl1 = CameraDL1Calibrator(**kwargs)
                dl1.calibrate(event)

                # Perform CHECM Charge Extraction
                area[ev] = event.dl1.tel[telid].image

            desc2 = "Fitting pixels"
            for pix in trange(self.n_pixels, desc=desc2):
                pixel_area = area[:, pix]
                if pix in self.dead.dead_pixels:
                    continue
                if self.calc_mean:
                    self.charge[fn, pix] = np.mean(pixel_area)
                    self.charge_err[fn, pix] = np.std(pixel_area)
                else:
                    if not self.fitter.apply(pixel_area):
                        self.log.warning("FN {} Pixel {} could not be fitted"
                                         .format(fn, pix))
                        self.charge.mask[fn, pix] = True
                        self.charge_err.mask[fn, pix] = True
                        continue
                    self.charge[fn, pix] = self.fitter.coeff['mean']
                    self.charge_err[fn, pix] = self.fitter.coeff['stddev']

        self.charge = np.ma.filled(self.dead.mask2d(self.charge))
        self.charge_err = np.ma.filled(self.dead.mask2d(self.charge_err))

    def finish(self):
        # Save figures
        output_dir = dirname(self.output_path)
        if not exists(output_dir):
            self.log.info("Creating directory: {}".format(output_dir))
            makedirs(output_dir)

        np.savez(self.output_path, charge=self.charge,
                 charge_error=self.charge_err, rundesc=self.rundesc_list)
        self.log.info("Numpy array saved to: {}".format(self.output_path))

    def get_next_file(self):
        for fr in self.file_looper.file_reader_list:
            yield fr


exe = ChargeVsRunExtractor()
exe.run()
