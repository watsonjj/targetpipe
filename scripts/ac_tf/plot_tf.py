import pandas as pd
from core import input_path, Plotter, plot_dir
from tf import *
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from IPython import embed
import pickle


class TFPlot(Plotter):
    def plot(self, x, y, label):
        # self.ax.plot(x, y, 'x-', mew=0.5, label=label)
        self.ax.plot(x, y, mew=0.5, label=label)

        self.ax.set_xlabel("ADC")
        self.ax.set_ylabel("Amplitude (mV)")
        self.ax.legend(loc="upper right")

    def add_legend(self):
        self.ax.legend(loc="upper right")


def main():
    pickle_path = join(plot_dir, "plot_tf.p")

    tf_list = [
        TFSamplingCell,
        TFStorageCell,
        TFStorageCellReduced,
        TFStorageCellReducedCompress,
        TFPChip
    ]

    tfs = {}

    desc = "Looping through TF list"
    for TF in tqdm(tf_list):
        tf_c = TF()
        adc_x, tf = tf_c._load_tf()
        tfs[tf_c.name] = (adc_x, tf)

    pickle.dump(tfs, open(pickle_path, "wb"))

    tfs = pickle.load(open(pickle_path, "rb"))

    r1 = [
        'TFSamplingCell',
        # 'TFStorageCell',
        # 'TFStorageCellReduced',
        # 'TFStorageCellReducedInt',
        # 'TFPChip'
    ]
    p_comparison = TFPlot(figure_name="tf_comparison")
    desc = "Generating plots"
    for name in tqdm(r1):
        p_tf = TFPlot(figure_name="tf_{}".format(name))
        x, tf = tfs[name]
        p_comparison.plot(x, tf[0], name)
        p_tf.plot(x, tf.T, "")
        p_tf.save()
    p_comparison.add_legend()
    p_comparison.save()


if __name__ == '__main__':
    main()
