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
# filepath = get_path(sys.argv[1])

source = hessio_event_source(filepath, max_events=1)
event = next(source)
tel = list(event.dl0.tels_with_data)[0]
pos_arr = np.array(event.inst.pixel_pos[tel])

n_pix = pos_arr.shape[1]

pos_arr = pos_arr[:, checm_pixel_id]
pos_arr[1] = -pos_arr[1]

fig = plt.figure(figsize=(13, 13))
ax = fig.add_subplot(111)
geom = CameraGeometry.guess(*event.inst.pixel_pos[tel],
                            event.inst.optical_foclen[tel])
camera = CameraDisplay(geom, ax=ax, image=np.zeros(2048))

for pix in range(n_pix):
    pos_x = pos_arr[0, pix]
    pos_y = pos_arr[1, pix]
    ax.text(pos_x, pos_y, pix, fontsize=3, color='w', ha='center')
    # print("[{0:.5g}, {1:.5g}],  # {2}".format(pos_x, pos_y, pix))

name = "checm_pixel_pos.npy"
path = join(realpath(dirname(__file__)), "../targetpipe/io", name)
np.save(path, pos_arr)
plt.show()
