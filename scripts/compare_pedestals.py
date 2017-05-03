from traitlets import Dict, List, Unicode
from ctapipe.core import Tool
from targetpipe.calib.camera.pedestal import PedestalSubtractor
import numpy as np


class PedestalComparer(Tool):
    name = "PedestalComparer"
    description = "Compare between two TargetCalib pedestals."

    p1_path = Unicode("", help="Path to an r0 file.").tag(config=True)
    p2_path = Unicode("", help="Path to an r1 file.").tag(config=True)

    aliases = Dict(dict(p1='PedestalComparer.p1_path',
                        p2='PedestalComparer.p2_path',
                        ))
    classes = List([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ps1 = None
        self.ps2 = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.ps1 = PedestalSubtractor(**kwargs, pedestal_path=self.p1_path)
        self.ps2 = PedestalSubtractor(**kwargs, pedestal_path=self.p2_path)

    def start(self):
        p1 = self.ps1.get_ped()
        p2 = self.ps2.get_ped()

        from IPython import embed
        embed()

        assert np.allclose(p1, p2)

    def finish(self):
        pass


if __name__ == '__main__':
    exe = PedestalComparer()
    exe.run()
