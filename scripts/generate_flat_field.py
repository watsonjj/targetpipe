from targetpipe.io.camera import Config
Config('checm')

from tqdm import tqdm, trange
from traitlets import Dict, List
import numpy as np
from os.path import realpath, join, dirname, exists
from os import makedirs

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import AverageWfPeakIntegrator
from ctapipe.image.waveform_cleaning import CHECMWaveformCleanerAverage
from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.io.pixels import Dead
from targetpipe.calib.camera.filter_wheel import FWCalibrator
from target_calib import CfMaker


class FFGenerator(Tool):
    name = "FFGenerator"
    description = "Generate Flat Fielding file"

    aliases = Dict(dict())
    classes = List([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reader = None
        self.dl0 = None
        self.dl1 = None
        self.fitter = None
        self.dead = None
        self.fw_calibrator = None

        self.cfmaker = None

        self.fw_pos = 2250

        directory = join(realpath(dirname(__file__)), "../adc2pe")
        self.output_path = join(directory, "ff.tcal")
        if not exists(directory):
            self.log.info("Creating directory: {}".format(directory))
            makedirs(directory)

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        filepath = '/Volumes/gct-jason/data/170319/linearity/linearity/Run03991_r1_pe.tio'
        # filepath = '/Volumes/gct-jason/data/170320/linearity/Run04164_r1_pe.tio'
        self.reader = TargetioFileReader(input_path=filepath, **kwargs)

        cleaner = CHECMWaveformCleanerAverage(**kwargs)
        extractor = AverageWfPeakIntegrator(**kwargs)
        self.dl0 = CameraDL0Reducer(**kwargs)
        self.dl1 = CameraDL1Calibrator(extractor=extractor,
                                       cleaner=cleaner,
                                       **kwargs)
        self.dead = Dead()
        self.fw_calibrator = FWCalibrator(**kwargs)

        self.cfmaker = CfMaker(32)

    def start(self):
        n_events = self.reader.num_events
        first_event = self.reader.get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        n_pixels, n_samples = first_event.r1.tel[telid].pe_samples[0].shape

        dl1 = np.zeros((n_events, n_pixels))
        ff = np.zeros(n_pixels)
        fw_illumination = self.fw_calibrator.get_illumination(self.fw_pos)

        source = self.reader.read()
        desc = "Looping through file"
        for event in tqdm(source, total=n_events, desc=desc):
            index = event.count
            self.dl0.reduce(event)
            self.dl1.calibrate(event)
            dl1[index] = event.dl1.tel[telid].image

        desc = "Fitting pixels"
        for pix in trange(n_pixels, desc=desc):
            if pix in self.dead.dead_pixels:
                continue
            charge = np.mean(dl1[:, pix])
            ff[pix] = fw_illumination/charge

        self.cfmaker.SetAll(ff.astype(np.float32))
        self.cfmaker.Save(self.output_path, False)
        self.log.info("FF tcal created: {}".format(self.output_path))
        self.cfmaker.Clear()

    def finish(self):
        pass


if __name__ == '__main__':
    exe = FFGenerator()
    exe.run()
