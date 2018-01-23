import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from core import pedestal_path, Plotter, plot_dir
from tf import TFSamplingCell, TFStorageCell, TFStorageCellPedestal, TFStorageCellPedestalZero
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from IPython import embed
import pickle
from os.path import join


class WaveformPlotter(Plotter):
    name = 'WaveformPlotter'

    def create(self, waveform, title, units):
        self.ax.plot(waveform, color='black')
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (ns)")
        self.ax.set_ylabel("Amplitude ({})".format(units))
        self.ax.xaxis.set_major_locator(MultipleLocator(16))


class HistPlotter(Plotter):
    name = 'HistPlotter'

    def add(self, data, label):
        data = data.ravel()

        mean = np.mean(data)
        stddev = np.std(data)
        N = data.size

        l = "{} (Mean = {:.2f}, Stddev = {:.2f}, N = {:.2e})".format(label, mean, stddev, N)
        self.ax.hist(data, bins=100, range=[-10, 10], label=l, alpha=0.7)

        self.ax.set_xlabel("Pedestal-Subtracted Residuals")
        self.ax.set_ylabel("N")
        # self.ax.xaxis.set_major_locator(MultipleLocator(16))

    def save(self, output_path=None):
        self.ax.legend(loc="upper left", fontsize=5)
        super().save(output_path)


def main():
    pickle_path = join(plot_dir, "plot_pedestal.p")

    store = pd.HDFStore(pedestal_path)
    df = store['df']

    tf_list = [TFSamplingCell, TFStorageCell, TFStorageCellPedestal, TFStorageCellPedestalZero]

    samples = {}

    desc = "Looping through TF list"
    samples["Pedestal Subtracted ADC"] = df['adc'].values
    for cls in tqdm(tf_list, desc=desc):
        tf = cls()
        df_cal = tf.calibrate(df)
        samples[tf.__class__.__name__] = df_cal['cal'].values

    pickle.dump(samples, open(pickle_path, "wb"))

    samples = pickle.load(open(pickle_path, "rb"))

    hist = HistPlotter(figure_name="pedestal_hist")
    desc = "Adding to hist"
    for name, array in tqdm(samples.items(), desc=desc):
        hist.add(array, name)
    hist.save()


if __name__ == '__main__':
    main()
