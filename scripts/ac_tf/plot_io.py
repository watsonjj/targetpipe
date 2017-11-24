import pandas as pd
from matplotlib.colors import LogNorm

from core import input_path, Plotter, plot_dir
from tf import *
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from IPython import embed
import pickle


class Residual2DPlot(Plotter):
    def plot(self, x, y):
        xbins = np.linspace(x.min(), x.max()+1, int(x.max() - x.min() + 2)//4) - 0.5
        _, _, _, h = self.ax.hist2d(x, y, bins=[xbins, 100], cmap='viridis', norm=LogNorm())

        lims = [
            np.min([self.ax.get_xlim(), self.ax.get_ylim()]),
            np.max([self.ax.get_xlim(), self.ax.get_ylim()]),
        ]
        self.ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)

        cbar = plt.colorbar(h)
        self.ax.set_xlabel("Input Amplitude (mV)")
        self.ax.set_ylabel("TF Calibrated Amplitude (mV)")
        cbar.set_label("Counts")
        self.ax.set_xlim(-300, 2600)
        self.ax.set_ylim(-300, 2600)

    def zoom(self):
        self.ax.set_xlim(-100, 100)


def main():
    pickle_path = join(plot_dir, "plot_io.p")

    store = pd.HDFStore(input_path)
    df = store['df']

    tf_list = [
        TFSamplingCell,
        TFStorageCell,
        # TFStorageCellReduced,
        # TFStorageCellReducedCompress,
        # TFPChip,
        # TFBest,
        # TFNothing,
        # TFTargetCalib
    ]

    io = {}

    desc = "Looping through TF list"
    for TF in tqdm(tf_list, desc=desc):
        tf = TF()
        df_cal = tf.calibrate(df)
        io[tf.name] = df_cal.loc[:, ['vped', 'cal']]

    pickle.dump(io, open(pickle_path, "wb"))

    io = pickle.load(open(pickle_path, "rb"))

    r1 = [
        'TFSamplingCell',
        'TFStorageCell',
        # 'TFStorageCellReduced',
        # 'TFStorageCellReducedCompress',
        # 'TFPChip',
        # 'TFBest',
        # 'TFNothing',
        # 'TFTargetCalib'
    ]
    desc = "Generating plots"
    for name in tqdm(r1, desc=desc):
        p_2d = Residual2DPlot(figure_name='residualio_{}'.format(name))
        p_2d.plot(io[name]['vped'], io[name]['cal'])
        p_2d.save()


if __name__ == '__main__':
    main()
