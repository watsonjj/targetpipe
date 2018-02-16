from tqdm import tqdm
from traitlets import Dict, List
import numpy as np

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import AverageWfPeakIntegrator
from ctapipe.image.waveform_cleaning import CHECMWaveformCleanerAverage
from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.fitting.chec import CHECMSPEFitter
from targetpipe.io.pixels import Dead


class FitComparer(Tool):
    name = "ADC2PEPlots"
    description = "Create plots related to adc2pe"

    aliases = Dict(dict(max_events='TargetioFileReader.max_events'
                        ))
    classes = List([TargetioFileReader,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reader_uncal = None
        self.reader_cal = None
        self.dl0 = None
        self.dl1 = None
        self.dead = None
        self.fitter_uncal = None
        self.fitter_cal = None

        self.n_pixels = None
        self.n_samples = None

        self.path_uncal = "/Volumes/gct-jason/data/170314/spe/Run00073_r1_adc.tio"
        self.path_cal = "/Volumes/gct-jason/data/170314/spe/Run00073_r1.tio"

        self.p_comparison = None
        self.p_tmspread = None
        self.p_dist = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.reader_uncal = TargetioFileReader(input_path=self.path_uncal,
                                               **kwargs)
        self.reader_cal = TargetioFileReader(input_path=self.path_cal,
                                             **kwargs)

        cleaner = CHECMWaveformCleanerAverage(**kwargs)
        extractor = AverageWfPeakIntegrator(**kwargs)
        self.dl0 = CameraDL0Reducer(**kwargs)
        self.dl1 = CameraDL1Calibrator(extractor=extractor,
                                       cleaner=cleaner,
                                       **kwargs)
        self.dead = Dead()

        self.fitter_uncal = CHECMSPEFitter(**kwargs)
        self.fitter_cal = CHECMSPEFitter(**kwargs)
        self.fitter_cal.range = [-3, 6]
        self.fitter_cal.initial = dict(norm=None,
                                       eped=0,
                                       eped_sigma=0.2,
                                       spe=1,
                                       spe_sigma=0.5,
                                       lambda_=0.2)

        first_event = self.reader_uncal.get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        r1 = first_event.r1.tel[telid].pe_samples[0]
        self.n_pixels, self.n_samples = r1.shape

    def start(self):
        n_events = self.reader_uncal.num_events

        dl1_uncal = np.zeros((n_events, self.n_pixels))
        dl1_cal = np.zeros((n_events, self.n_pixels))

        source_uncal = self.reader_uncal.read()
        desc = 'Looping through events: uncal'
        for event in tqdm(source_uncal, total=n_events, desc=desc):
            ev = event.count
            self.dl0.reduce(event)
            self.dl1.calibrate(event)
            dl1_uncal[ev] = event.dl1.tel[0].image[0]

        source_cal = self.reader_cal.read()
        desc = 'Looping through events: cal'
        for event in tqdm(source_cal, total=n_events, desc=desc):
            ev = event.count
            self.dl0.reduce(event)
            self.dl1.calibrate(event)
            dl1_cal[ev] = event.dl1.tel[0].image[0]

        # np.save("/Users/Jason/Downloads/dl1_uncal.npy", dl1_uncal)
        # np.save("/Users/Jason/Downloads/dl1_cal.npy", dl1_cal)
        #
        # dl1_uncal = np.load("/Users/Jason/Downloads/dl1_uncal.npy")
        # dl1_cal = np.load("/Users/Jason/Downloads/dl1_cal.npy")

        for pix in range(self.n_pixels):
            if not self.fitter_uncal.apply(dl1_uncal[:, pix]):
                self.log.warning("Pixel {} couldn't be fit".format(pix))
                continue
            if not self.fitter_cal.apply(dl1_cal[:, pix]):
                self.log.warning("Pixel {} couldn't be fit".format(pix))
                continue

            gain_uncal = self.fitter_uncal.coeff['spe']
            gain_cal = self.fitter_cal.coeff['spe']
            lambda_uncal = self.fitter_uncal.coeff['lambda_']
            lambda_cal = self.fitter_cal.coeff['lambda_']

            print("Pixel {}: Gain Uncal {:.3} Gain Cal {:.3} "
                  "Lambda Uncal {:.3} Lambda Cal {:.3}"
                  .format(pix, gain_uncal, gain_cal, lambda_uncal, lambda_cal))

    def finish(self):
        pass


if __name__ == '__main__':
    exe = FitComparer()
    exe.run()
