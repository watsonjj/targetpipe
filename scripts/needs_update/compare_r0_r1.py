from traitlets import Dict, List, Unicode
from ctapipe.core import Tool
from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.calib.camera.r1 import TargetioR1Calibrator
from tqdm import trange
import numpy as np
from IPython import embed


class R0R1Comparer(Tool):
    name = "R0R1Comparer"
    description = "Compare between an r0 and r1 file to check the " \
                  "calibration applied is identical."

    r0_path = Unicode("", help="Path to an r0 file.").tag(config=True)
    r1_path = Unicode("", help="Path to an r1 file.").tag(config=True)

    aliases = Dict(dict(r0='R0R1Comparer.r0_path',
                        r1='R0R1Comparer.r1_path',
                        max_events='TargetioFileReader.max_events',
                        ped='TargetioR1Calibrator.pedestal_path',
                        tf='TargetioR1Calibrator.tf_path',
                        pe='TargetioR1Calibrator.pe_path',
                        ))
    classes = List([TargetioFileReader,
                    TargetioR1Calibrator,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reader_r0 = None
        self.reader_r1 = None
        self.r1 = None
        self.dl0 = None
        self.dl1 = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        if not self.r0_path or not self.r1_path:
            raise FileNotFoundError("Both r0 and r1 paths need to be set")

        self.reader_r0 = TargetioFileReader(**kwargs, input_path=self.r0_path)
        self.reader_r1 = TargetioFileReader(**kwargs, input_path=self.r1_path)
        self.r1 = TargetioR1Calibrator(**kwargs)

        assert self.reader_r0.num_events == self.reader_r1.num_events

    def start(self):
        n_events = self.reader_r0.num_events
        source_r0 = self.reader_r0.read()
        source_r1 = self.reader_r1.read()
        desc = "Looping through both files"
        for ev in trange(n_events, desc=desc):
            event_r0 = self.reader_r0.get_event(ev)
            event_r1 = self.reader_r1.get_event(ev)

            self.r1.calibrate(event_r0)
            self.r1.calibrate(event_r1)

            samples_r0 = event_r0.r1.tel[0].pe_samples
            samples_r1 = event_r1.r1.tel[0].pe_samples

            np.testing.assert_almost_equal(samples_r0, samples_r1, 1)

        self.log.info("All events match!")

    def finish(self):
        pass


if __name__ == '__main__':
    exe = R0R1Comparer()
    exe.run()
