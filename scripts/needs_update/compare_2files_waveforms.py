from traitlets import Dict, List, Unicode
from ctapipe.core import Tool
from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.calib.camera.r1 import TargetioR1Calibrator
from tqdm import trange
import numpy as np
from IPython import embed


class Comparer(Tool):
    name = "Comparer"
    description = "Compare between two files to check the " \
                  "waveforms are identical."

    path1 = Unicode("", help="Path to a first file.").tag(config=True)
    path2 = Unicode("", help="Path to a second file.").tag(config=True)

    aliases = Dict(dict(p1='Comparer.path1',
                        p2='Comparer.path2',
                        max_events='TargetioFileReader.max_events',
                        ))
    classes = List([TargetioFileReader,
                    TargetioR1Calibrator,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reader1 = None
        self.reader2 = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        if not self.path1 or not self.path2:
            raise FileNotFoundError("Both paths need to be set")

        self.reader1 = TargetioFileReader(**kwargs, input_path=self.path1)
        self.reader2 = TargetioFileReader(**kwargs, input_path=self.path2)

        assert self.reader1.num_events == self.reader2.num_events

    def start(self):
        n_events = self.reader1.num_events
        source1 = self.reader1.read()
        source2 = self.reader2.read()
        desc = "Looping through both files"
        for ev in trange(n_events, desc=desc):
            event1 = self.reader1.get_event(ev)
            event2 = self.reader2.get_event(ev)

            samples1 = event1.r1.tel[0].pe_samples
            samples2 = event2.r1.tel[0].pe_samples
            if (samples1==0).all():
                samples1 = event1.r0.tel[0].adc_samples
            if (samples2==0).all():
                samples2 = event1.r0.tel[0].adc_samples

            np.testing.assert_almost_equal(samples1, samples2, 1)

        self.log.info("All events match!")

    def finish(self):
        pass


if __name__ == '__main__':
    exe = Comparer()
    exe.run()
