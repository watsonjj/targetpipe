import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os.path import dirname, join
from tqdm import tqdm
from core import pix, pix_path as input_path, Plotter
from IPython import embed


class scatterplot(Plotter):
    def plot(self, x, y, c, xl, yl, cl):
        p = self.ax.scatter(x, y, c=c, cmap='viridis')
        cb = self.fig.colorbar(p)
        self.ax.set_xlabel(xl)
        self.ax.set_ylabel(yl)
        cb.set_label(cl)


class rmsdist(Plotter):
    def plot(self, x, y, yerr, xl, yl):
        (_, caps, _) = self.ax.errorbar(x, y, yerr=yerr, fmt='o', mew=0.5, alpha=0.8, markersize=3, capsize=3, ecolor='gray')

        for cap in caps:
            cap.set_markeredgewidth(1)

        self.ax.set_xlabel(xl)
        self.ax.set_ylabel(yl)


def main():
    store = pd.HDFStore(input_path)
    df = store['df']
    df['samplingcell'] = (df['fci'] + df['sample']) % 64
    df['incorrectcell'] = (df['fci'] + df['sample']) % 4096

    tqdm.pandas(desc="Performing pandas operation")
    df_std_sampling = df.groupby(['vped', 'samplingcell']).std().reset_index()
    df_std_cell = df.groupby(['vped', 'cell']).std().reset_index()
    df_std_incell = df.groupby(['vped', 'incorrectcell']).std().reset_index()

    # p_sampling = scatterplot(figure_name='rms_samplingcell')
    # p_sampling.plot(df_std_sampling['samplingcell'], df_std_sampling['vped'], df_std_sampling['adc'], "Sampling Cell", "Amplitude (mV)", "ADC RMS")
    # p_sampling.save()
    #
    # p_cell = scatterplot(figure_name='rms_cell')
    # p_cell.plot(df_std_cell['cell'], df_std_cell['vped'], df_std_cell['adc'], "Storage Cell", "Amplitude (mV)", "ADC RMS")
    # p_cell.save()
    #
    # p_incell = scatterplot(figure_name='rms_incell')
    # p_incell.plot(df_std_incell['incorrectcell'], df_std_incell['vped'], df_std_incell['adc'], "Incorrect Sampling Cell", "Amplitude (mV)", "ADC RMS")
    # p_incell.save()

    p_rmsdist_sampling = rmsdist(figure_name='rmsdist_samplingcell')
    g = df_std_sampling.groupby(['samplingcell'])
    mean = g.mean().reset_index()
    min_ = g.min()
    max_ = g.max()
    x = mean['samplingcell']
    y = mean['adc']
    yerr = np.vstack([min_['adc'], max_['adc']])
    p_rmsdist_sampling.plot(x, y, yerr, "Sampling Cell", "ADC RMS")
    p_rmsdist_sampling.save()

    p_rmsdist_cell = rmsdist(figure_name='rmsdist_cell')
    g = df_std_cell.groupby(['cell'])
    mean = g.mean().reset_index()
    min_ = g.min()
    max_ = g.max()
    x = mean['cell']
    y = mean['adc']
    yerr = np.vstack([min_['adc'], max_['adc']])
    p_rmsdist_cell.plot(x, y, yerr, "Storage Cell", "ADC RMS")
    p_rmsdist_cell.save()

    p_rmsdist_incell = rmsdist(figure_name='rmsdist_incell')
    g = df_std_incell.groupby(['incorrectcell'])
    mean = g.mean().reset_index()
    min_ = g.min()
    max_ = g.max()
    x = mean['incorrectcell']
    y = mean['adc']
    yerr = np.vstack([min_['adc'], max_['adc']])
    p_rmsdist_incell.plot(x, y, yerr, "Incorrect Storage Cell", "ADC RMS")
    p_rmsdist_incell.save()



if __name__ == '__main__':
    main()
