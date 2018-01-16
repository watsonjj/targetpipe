import pandas as pd
from core import input_path, Plotter, plot_dir
from tf import TF, child_subclasses
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from IPython import embed
import pickle
from os.path import join


class TFPlot(Plotter):
    def plot(self, x, y, label):
        self.ax.plot(x, y, 'x-', mew=0.5, label=label)

        self.ax.set_xlabel("Input Pulse Amplitude (mV)")
        self.ax.set_ylabel("ADC")
        self.ax.legend(loc="upper right")

    def add_legend(self):
        self.ax.legend(loc="upper right")


def main():
    pickle_path = join(plot_dir, "plot_tfinput.p")

    store = pd.HDFStore(input_path)
    df = store['df']

    tf_list = child_subclasses(TF)

    tfs = {}

    # desc = "Looping through TF list"
    # for cls in tqdm(tf_list):
    #     tf_c = cls()
    #     vped, adc = tf_c.get_tfinput(df)
    #     tfs[tf_c.__class__.__name__] = (vped, adc)
    #
    # pickle.dump(tfs, open(pickle_path, "wb"))

    tfs = pickle.load(open(pickle_path, "rb"))

    r1 = [
        'TFSamplingCell',
        'TFStorageCell',
        'TFStorageCellExp',
    ]
    p_comparison = TFPlot(figure_name="tfinput_comparison")
    desc = "Generating plots"
    for name in tqdm(r1):
        p_tf = TFPlot(figure_name="tfinput_{}".format(name))
        x, adc = tfs[name]
        p_comparison.plot(x, adc[0], name)
        y1 = adc.T
        y2 = np.gradient(adc.T[:, 0], x)
        # y1 = y1 / y1.max()
        # y2 = y2 / y2.max()
        p_tf.plot(x, y1, "")
        # p_tf.ax.set_xlim(-20, 125)
        # p_tf.ax.set_ylim(-20, 250)
        # p_tf.plot(x, y2, "")
        # p_tf.ax.set_xlim(-20, 125)
        # p_tf.ax.set_ylim(1, 2.5)
        p_tf.save()
    p_comparison.add_legend()
    p_comparison.save()


if __name__ == '__main__':
    main()
