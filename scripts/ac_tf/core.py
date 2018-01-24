from os.path import join
from targetpipe.plots.official import Plotter as OriginalPlotter
from matplotlib import pyplot as plt

pix = 0
directory = "/Volumes/gct-jason/data_checs/tf/testing/"
core_path = join(directory, "Amplitude_input.csv")
pix_dir = join(directory, "p{}".format(pix))
plot_dir = join(pix_dir, "plots")
input_path = join(pix_dir, "input.h5")
pedestal_path = join(pix_dir, "pedestal.h5")
tf_path = "/Users/Jason/Software/CHECDevelopment/CHECS/Operation/SN0074_tf.tcal"


class Plotter(OriginalPlotter):
    def __init__(self, figure_name):
        type = 'paper'
        base_dir = plot_dir
        script = ""
        super().__init__(type, base_dir, "", figure_name)