import pandas as pd
from core import input_path, Plotter, plot_dir
from tf import *
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from IPython import embed
import pickle


class HitsPlotter(Plotter):
    def plot(self, value, vpeds):
        n_blks, n_bps = value.shape
        hits = np.ma.masked_where(value == 0, value)

        im = self.ax.pcolor(hits, cmap="viridis", edgecolors='white', linewidths=0)

        self.ax.set_xticks(np.arange(vpeds.size)+0.5)
        self.ax.set_xticklabels(vpeds, rotation='vertical', fontsize=5)
        self.ax.tick_params(axis='x', which='minor', length=0)

        cbar = self.fig.colorbar(im)
        self.ax.patch.set(hatch='xx')
        self.ax.set_xlabel("Input Amplitude (mV)")
        self.ax.set_ylabel("Cell")
        cbar.set_label("Hits")

        # self.ax.set_ylim(110, 120)
        plt.show()

def main():
    pickle_path = join(plot_dir, "plot_hits.p")

    store = pd.HDFStore(input_path)
    df = store['df']

    vpeds = np.unique(df['vped'])

    # tf_list = [
    #     TFSamplingCell,
    #     TFStorageCell,
    #     TFStorageCellReduced,
    #     TFStorageCellReducedInt,
    #     TFPChip
    # ]
    #
    # hits_dict = {}
    #
    # desc = "Looping through TF list"
    # for TF in tqdm(tf_list):
    #     tf_c = TF()
    #     hits = tf_c.get_hits(df)
    #     hits_dict[tf_c.name] = hits
    #
    # pickle.dump(hits_dict, open(pickle_path, "wb"))

    hits_dict = pickle.load(open(pickle_path, "rb"))

    r1 = ['TFSamplingCell', 'TFStorageCell']
    desc = "Generating plots"
    for name in tqdm(r1):
        p_hits = HitsPlotter(figure_name="hits_{}".format(name))
        p_hits.plot(hits_dict[name], vpeds)
        p_hits.save()


if __name__ == '__main__':
    main()
