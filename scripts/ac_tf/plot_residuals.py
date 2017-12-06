import pandas as pd
from matplotlib.colors import LogNorm

from core import input_path, Plotter, plot_dir
from tf import *
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from IPython import embed
import pickle


def rmse(true, measured):
    n = measured.count()
    return np.sqrt(np.sum(np.power(true - measured, 2)) / n)/np.abs(true)


def rmse_df(row):
    row['rmse'] = rmse(row['vped'].iloc[0], row['cal'])
    return row


class ResidualPlot(Plotter):
    def plot(self, df, label):
        x = df['vped']
        y = df['rmse']
        self.ax.plot(x, y, 'x-', mew=0.5, label=label)
        self.ax.set_yscale("log", nonposy='clip')

        self.ax.set_xlabel("Input Amplitude (mV)")
        self.ax.set_ylabel("Voltage Resolution")
        self.ax.legend(loc="upper right")

    def zoom(self):
        self.ax.set_xlim(-100, 100)


def main():
    pickle_path = join(plot_dir, "plot_residuals.p")

    # store = pd.HDFStore(input_path)
    # df = store['df']
    #
    # tf_list = [
    #     TFSamplingCell,
    #     TFStorageCell,
    #     TFStorageCellReduced,
    #     TFStorageCellReducedCompress,
    #     TFPChip,
    #     TFBest,
    #     TFBestCompress,
    #     TFNothing,
    #     TFTargetCalib
    # ]
    #
    # residuals = {}
    #
    # desc = "Looping through TF list"
    # for TF in tqdm(tf_list, desc=desc):
    #     tf = TF()
    #     df_cal = tf.calibrate(df)
    #     tqdm.pandas(desc="Obtaining residuals")
    #     df_rmse = df_cal.groupby('vped').progress_apply(rmse_df).reset_index()
    #     df_residuals = df_rmse.loc[:, ['vped', 'rmse']].groupby('vped').first().reset_index()
    #     residuals[tf.name] = df_residuals
    #
    # pickle.dump(residuals, open(pickle_path, "wb"))

    residuals = pickle.load(open(pickle_path, "rb"))

    r1 = [
        'TFStorageCell',
        'TFStorageCellReduced',
        'TFPChip',
        'TFBest',
        'TFNothing',
    ]
    p_res = ResidualPlot(figure_name="residual_1")
    desc = "Generating plots"
    for name in tqdm(r1, desc=desc):
        p_res.plot(residuals[name], name)
    p_res.save()

    r2 = [
        'TFSamplingCell',
        'TFTargetCalib'
    ]
    p_res = ResidualPlot(figure_name="residual_2")
    desc = "Generating plots"
    for name in tqdm(r2, desc=desc):
        p_res.plot(residuals[name], name)
    p_res.save()

    r3 = [
        'TFBest',
        'TFNothing'
    ]
    p_res = ResidualPlot(figure_name="residual_3")
    desc = "Generating plots"
    for name in tqdm(r3, desc=desc):
        p_res.plot(residuals[name], name)
    p_res.save()

    r4 = [
        'TFBest',
        'TFBestCompress',
        'TFStorageCellReduced',
        'TFStorageCellReducedCompress'
    ]
    p_res = ResidualPlot(figure_name="residual_4")
    desc = "Generating plots"
    for name in tqdm(r4, desc=desc):
        p_res.plot(residuals[name], name)
    p_res.save()

    r5 = [
        'TFSamplingCell',
        'TFStorageCell',
    ]
    p_res = ResidualPlot(figure_name="residual_5")
    desc = "Generating plots"
    for name in tqdm(r5, desc=desc):
        p_res.plot(residuals[name], name)
    p_res.save()

    r6 = [
        'TFSamplingCell'
    ]
    p_res = ResidualPlot(figure_name="residual_6")
    desc = "Generating plots"
    for name in tqdm(r6, desc=desc):
        p_res.plot(residuals[name], name)
    p_res.save()

    r7 = [
        'TFSamplingCell',
        'TFStorageCell',
        'TFStorageCellReducedCompress'
    ]
    p_res = ResidualPlot(figure_name="residual_7")
    desc = "Generating plots"
    for name in tqdm(r7, desc=desc):
        p_res.plot(residuals[name], name)
    p_res.save()

    p_res = ResidualPlot(figure_name="residual_1z")
    for name in r1:
        p_res.plot(residuals[name], name)
    p_res.ax.set_xlim(-100, 100)
    p_res.ax.set_ylim(8E-3, 3)
    p_res.save()

    p_res = ResidualPlot(figure_name="residual_6z")
    desc = "Generating plots"
    for name in tqdm(r6, desc=desc):
        p_res.plot(residuals[name], name)
    p_res.ax.set_xlim(-100, 100)
    p_res.ax.set_ylim(8E-3, 3)
    p_res.save()

    p_res = ResidualPlot(figure_name="residual_7z")
    desc = "Generating plots"
    for name in tqdm(r7, desc=desc):
        p_res.plot(residuals[name], name)
    p_res.ax.set_xlim(-100, 100)
    p_res.ax.set_ylim(8E-3, 3)
    p_res.save()


if __name__ == '__main__':
    main()
