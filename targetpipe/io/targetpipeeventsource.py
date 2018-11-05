from os.path import join, dirname, realpath
import numpy as np
from ctapipe.io.targetioeventsource import TargetIOEventSource


class TargetpipeEventSource(TargetIOEventSource):
    def _update_container(self):
        super()._update_container()
        data = self._data
        chec_tel = 0

        # CHEC-M pulse shape
        if self.camera_config.GetVersion() == "1.0.0":
            path = join(dirname(realpath(__file__)),
                        'checm_reference_pulse.npz')
            file = np.load(path)
            data.mc.tel[chec_tel].reference_pulse_shape = file['refshape']
            data.mc.tel[chec_tel].meta['refstep'] = file['refstep']
            data.mc.tel[chec_tel].time_slice = file['time_slice']
