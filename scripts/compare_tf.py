from traitlets import Dict, List, Unicode
from ctapipe.core import Tool
from targetpipe.calib.camera.tf import TFApplier
import numpy as np
from IPython import embed


class TfComparer(Tool):
    name = "TfComparer"
    description = "Compare between two TargetCalib TFs."

    tf1_path = Unicode("", help="Path to the first tf file.").tag(config=True)
    tf2_path = Unicode("", help="Path to the second tf file.").tag(config=True)

    aliases = Dict(dict(tf1='TfComparer.tf1_path',
                        tf2='TfComparer.tf2_path',
                        ))
    classes = List([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tfa1 = None
        self.tfa2 = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.tfa1 = TFApplier(**kwargs, tf_path=self.tf1_path)
        self.tfa2 = TFApplier(**kwargs, tf_path=self.tf2_path)

    def start(self):
        tf1 = self.tfa1.get_tf()[0]
        tf2 = self.tfa2.get_tf()[0]

        assert np.allclose(tf1, tf2)

    def finish(self):
        pass


if __name__ == '__main__':
    exe = TfComparer()
    exe.run()
