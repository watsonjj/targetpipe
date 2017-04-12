import numpy as np
from matplotlib import pyplot as plt
from os.path import realpath, join, dirname
from ctapipe.utils.datasets import get_path
from ctapipe.io.hessio import hessio_event_source
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from targetpipe.io.pixels import checm_pixel_id


filepath = get_path("/Users/Jason/Software/outputs/sim_telarray/meudon_gamma/"
                    "simtel_runmeudon_gamma_30tel_30deg_19.gz")

source = hessio_event_source(filepath, max_events=1)
event = next(source)
telid = list(event.dl0.tels_with_data)[0]

refshape = event.mc.tel[telid].reference_pulse_shape
refstep = event.mc.tel[telid].meta['refstep']
time_slice = event.mc.tel[telid].time_slice

name = "checm_reference_pulse.npz"
path = join(realpath(dirname(__file__)), "../targetpipe/io", name)
np.savez(path, refshape=refshape, refstep=refstep, time_slice=time_slice)
plt.show()
