from os import makedirs
from os.path import join, exists

import numpy as np
from tqdm import tqdm
from traitlets import Dict, List

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import AverageWfPeakIntegrator
from ctapipe.image.waveform_cleaning import CHECMWaveformCleanerLocal
from ctapipe.io.eventfilereader import EventFileReaderFactory
from targetpipe.fitting.chec import CHECMSPEFitter
from targetpipe.io.pixels import Dead


class ADC2PEResidualsExtractor(Tool):
    name = "ADC2PEResidualsExtractor"
    description = "Extract values used to show the residuals of the adc2pe " \
                  "calibration into a numpy array"

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        pe='CameraR1CalibratorFactory.adc2pe_path',
                        ))
    classes = List([EventFileReaderFactory,
                    CameraR1CalibratorFactory,
                    CHECMSPEFitter,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reader = None
        self.r1 = None
        self.dl0 = None
        self.cleaner = None
        self.extractor = None
        self.dl1 = None

        self.fitter = None
        self.dead = None

        self.output_dir = None

        self.spe = None
        self.spe_sigma = None
        self.hist = None
        self.edges = None
        self.between = None

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
        self.cleaner = CHECMWaveformCleanerLocal(**kwargs)
        self.extractor = AverageWfPeakIntegrator(**kwargs)
        self.dl0 = CameraDL0Reducer(**kwargs)
        self.dl1 = CameraDL1Calibrator(extractor=self.extractor,
                                       cleaner=self.cleaner,
                                       **kwargs)

        self.fitter = CHECMSPEFitter(**kwargs)
        # self.fitter.nbins = 60
        self.fitter.range = [-3, 6]
        self.fitter.initial = dict(norm=None,
                                   eped=0,
                                   eped_sigma=0.2,
                                   spe=1,
                                   spe_sigma=0.5,
                                   lambda_=0.2)

        self.dead = Dead()

        self.output_dir = join(self.reader.output_directory,
                               "extract_adc2pe_residuals")
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
        self.spe = np.ma.zeros(n_pixels)
        self.spe.mask = np.zeros(self.spe.shape, dtype=np.bool)
        self.spe_sigma = np.zeros(n_pixels)
        self.hist = np.zeros((n_pixels, self.fitter.nbins))
        self.edges = np.zeros((n_pixels, self.fitter.nbins+1))
        self.between = np.zeros((n_pixels, self.fitter.nbins))

        source = self.reader.read()
        desc = "Looping through file"
        with tqdm(total=n_events, desc=desc) as pbar:
            for event in source:
                pbar.update(1)
                index = event.count

                self.r1.calibrate(event)
                self.dl0.reduce(event)
                self.dl1.calibrate(event)

                # Perform CHECM Charge Extraction
                peak_area = event.dl1.tel[telid].image

                area[index] = peak_area

        desc = "Fitting pixels"
        with tqdm(total=n_pixels, desc=desc) as pbar:
            for pix in range(n_pixels):
                pbar.update(1)
                if not self.fitter.apply(area[:, pix]):
                    self.log.warning("Pixel {} couldn't be fit".format(pix))
                    self.spe.mask[pix] = True
                    continue
                self.spe[pix] = self.fitter.coeff['spe']
                self.spe_sigma[pix] = self.fitter.coeff['spe_sigma']
                self.hist[pix] = self.fitter.hist
                self.edges[pix] = self.fitter.edges
                self.between[pix] = self.fitter.between

    def finish(self):
        output_path = join(self.output_dir, "adc2pe_residuals.npz")
        np.savez(output_path,
                 spe=np.ma.filled(self.spe, 0),
                 spe_sigma=self.spe_sigma,
                 hist=self.hist,
                 edges=self.edges,
                 between=self.between)
        self.log.info("Created numpy array: {}".format(output_path))

exe = ADC2PEResidualsExtractor()
exe.run()
