from os.path import join
from targetpipe.plots.official import ThesisPlotter
from matplotlib import pyplot as plt

pix = 0
directory = "/Volumes/gct-jason/data_checs/tf/testing/"
core_path = join(directory, "Amplitude_input.csv")
pix_dir = join(directory, "p{}".format(pix))
plot_dir = join(pix_dir, "plots")
input_path = join(pix_dir, "input.h5")
tf_path = "/Users/Jason/Downloads/SN0074/TF_File_v4.tcal"


class Plotter(ThesisPlotter):
    def __init__(self, config=None, tool=None, **kwargs):
        super().__init__(config, tool, **kwargs)
        self.base_dir = plot_dir

    def create_figure(self):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        return fig, ax
