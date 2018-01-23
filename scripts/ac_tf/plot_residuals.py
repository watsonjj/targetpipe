import pandas as pd
from matplotlib.colors import LogNorm
from core import input_path, Plotter, plot_dir
from tf import *#TF, child_subclasses
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from IPython import embed
import pickle
from os.path import join


def rmse(true, measured):
    n = measured.count()
    return np.sqrt(np.sum(np.power(true - measured, 2)) / n)/np.abs(true)


def rmseabs(true, measured):
    n = measured.count()
    return np.sqrt(np.sum(np.power(true - measured, 2)) / n)


def rmse_df(row):
    row['rmse'] = rmse(row['vped'].iloc[0], row['cal'])
    row['rmseabs'] = rmseabs(row['vped'].iloc[0], row['cal'])
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


class AbsResidualPlot(Plotter):
    def plot(self, df, label):
        x = df['vped']
        y = df['rmseabs']
        self.ax.plot(x, y, 'x-', mew=0.5, label=label)
        self.ax.set_yscale("log", nonposy='clip')

        self.ax.set_xlabel("Input Amplitude (mV)")
        self.ax.set_ylabel("Absolute Voltage Resolution (mV)")
        self.ax.legend(loc="upper right")

    def zoom(self):
        self.ax.set_xlim(-100, 100)


def main():
    pickle_path = join(plot_dir, "plot_residuals.p")

    store = pd.HDFStore(input_path)
    df = store['df']
    # df = df.loc[(df['vped'] < 25) & (df['vped'] > -50)]
    store = pd.HDFStore(pedestal_path)
    df = pd.concat([df, store['df']], ignore_index=True)

    # tf_list = child_subclasses(TF)
    tf_list = [TFSamplingCell, TFStorageCell, TFStorageCellPedestal, TFStorageCellPedestalZero]

    residuals = {}

    desc = "Looping through TF list"
    for cls in tqdm(tf_list, desc=desc):
        tf = cls()
        df_cal = tf.calibrate(df)
        tqdm.pandas(desc="Obtaining residuals")
        df_rmse = df_cal.groupby('vped').progress_apply(rmse_df).reset_index()
        df_residuals = df_rmse.loc[:, ['vped', 'rmse', 'rmseabs']].groupby('vped').first().reset_index()
        residuals[tf.__class__.__name__] = df_residuals

    pickle.dump(residuals, open(pickle_path, "wb"))

    residuals = pickle.load(open(pickle_path, "rb"))

    r= [
        'TFSamplingCell',
        'TFStorageCell',
        'TFStorageCellPedestal',
        'TFStorageCellPedestalZero'
    ]
    p_res = AbsResidualPlot(figure_name="absresidual_p")
    desc = "Generating plots"
    for name in tqdm(r, desc=desc):
        p_res.plot(residuals[name], name)
    p_res.save()

    p_res = AbsResidualPlot(figure_name="absresidual_pz")
    desc = "Generating plots"
    for name in tqdm(r, desc=desc):
        p_res.plot(residuals[name], name)
    p_res.ax.set_xlim(-100, 100)
    p_res.ax.set_ylim(0.8, 6)
    p_res.save()

    pe = 4
    for name, res in residuals.items():
        print("___", name)
        val = res['vped'].values
        i = np.where((val > 0) & (val == val[val > 0].min()))[0][0]
        x = val[i]
        y = res['rmseabs'].values[i]
        print("{:.3} mV resolution @ {} mV".format(y, x))
        print("{:.3} p.e. resolution @ {} p.e.".format(y/pe, x/pe))

        i = np.argmin(np.absolute(val - 1000))
        x = val[i]
        y = res['rmseabs'].values[i]
        print("{:.3} mV resolution @ {} mV".format(y, x))
        print("{:.3} p.e. resolution @ {} p.e.".format(y/pe, x/pe))

        i = np.argmin(np.absolute(val - 0))
        x = val[i]
        y = res['rmseabs'].values[i]
        print("{:.3} mV resolution @ {} mV".format(y, x))
        print("{:.3} p.e. resolution @ {} p.e.".format(y / pe, x / pe))
    #
    # r = [
    #     'TFSamplingCell',
    # ]
    # p_res = ResidualPlot(figure_name="residual_0")
    # desc = "Generating plots"
    # for name in tqdm(r, desc=desc):
    #     p_res.plot(residuals[name], name)
    # p_res.save()
    #
    # p_res = ResidualPlot(figure_name="residual_0z")
    # desc = "Generating plots"
    # for name in tqdm(r, desc=desc):
    #     p_res.plot(residuals[name], name)
    # p_res.ax.set_xlim(-100, 100)
    # p_res.ax.set_ylim(8E-3, 0.5)
    # p_res.save()
    #
    # r = [
    #     'TFSamplingCell',
    # ]
    # p_res = AbsResidualPlot(figure_name="absresidual_0")
    # desc = "Generating plots"
    # for name in tqdm(r, desc=desc):
    #     p_res.plot(residuals[name], name)
    # p_res.save()
    #
    # p_res = AbsResidualPlot(figure_name="absresidual_0z")
    # desc = "Generating plots"
    # for name in tqdm(r, desc=desc):
    #     p_res.plot(residuals[name], name)
    # p_res.ax.set_xlim(-100, 100)
    # p_res.ax.set_ylim(1.5, 6)
    # p_res.save()
    #
    # r = [
    #     'TFSamplingCell',
    #     'TFSamplingCellPerfect',
    #     'TFStorageCell_60',
    #     'TFStorageCell_150',
    #     'TFStorageCell_300',
    #     'TFStorageCell_500',
    #     'TFStorageCell_1000',
    #     'TFStorageCellExp_60',
    #     'TFStorageCellExp_150'
    # ]
    # p_res = ResidualPlot(figure_name="residual_1")
    # desc = "Generating plots"
    # for name in tqdm(r, desc=desc):
    #     p_res.plot(residuals[name], name)
    # p_res.save()
    #
    # p_res = ResidualPlot(figure_name="residual_1z")
    # desc = "Generating plots"
    # for name in tqdm(r, desc=desc):
    #     p_res.plot(residuals[name], name)
    # p_res.ax.set_xlim(-100, 100)
    # p_res.ax.set_ylim(8E-3, 0.5)
    # p_res.save()
    #
    # r = [
    #     'TFSamplingCell',
    #     'TFSamplingCellPerfect',
    # ]
    # p_res = ResidualPlot(figure_name="residual_2")
    # desc = "Generating plots"
    # for name in tqdm(r, desc=desc):
    #     p_res.plot(residuals[name], name)
    # p_res.save()
    #
    # p_res = ResidualPlot(figure_name="residual_2z")
    # desc = "Generating plots"
    # for name in tqdm(r, desc=desc):
    #     p_res.plot(residuals[name], name)
    # p_res.ax.set_xlim(-100, 100)
    # p_res.ax.set_ylim(8E-3, 0.5)
    # p_res.save()
    #
    # r = [
    #     'TFStorageCell_60',
    #     'TFStorageCell_150',
    #     'TFStorageCell_300',
    #     'TFStorageCell_500',
    #     'TFStorageCell_1000',
    # ]
    # p_res = ResidualPlot(figure_name="residual_3")
    # desc = "Generating plots"
    # for name in tqdm(r, desc=desc):
    #     p_res.plot(residuals[name], name)
    # p_res.save()
    #
    # p_res = ResidualPlot(figure_name="residual_3z")
    # desc = "Generating plots"
    # for name in tqdm(r, desc=desc):
    #     p_res.plot(residuals[name], name)
    # p_res.ax.set_xlim(-100, 100)
    # p_res.ax.set_ylim(8E-3, 0.5)
    # p_res.save()
    #
    # r = [
    #     'TFSamplingCell',
    #     'TFStorageCell_60',
    #     'TFStorageCell_150',
    #     'TFStorageCell_300',
    #     'TFStorageCellExp_60',
    #     'TFStorageCellExp_150'
    # ]
    # p_res = ResidualPlot(figure_name="residual_4")
    # desc = "Generating plots"
    # for name in tqdm(r, desc=desc):
    #     p_res.plot(residuals[name], name)
    # p_res.save()
    #
    # p_res = ResidualPlot(figure_name="residual_4z")
    # desc = "Generating plots"
    # for name in tqdm(r, desc=desc):
    #     p_res.plot(residuals[name], name)
    # p_res.ax.set_xlim(-100, 100)
    # p_res.ax.set_ylim(8E-3, 0.5)
    # p_res.save()
    #
    # r = [
    #     'TFSamplingCell',
    #     'TFStorageCell_60',
    #     'TFStorageCell_150',
    #     'TFStorageCell_300',
    #     'TFStorageCellExp_60',
    #     'TFStorageCellExp_150'
    # ]
    # p_res = AbsResidualPlot(figure_name="absresidual_4")
    # desc = "Generating plots"
    # for name in tqdm(r, desc=desc):
    #     p_res.plot(residuals[name], name)
    # p_res.save()
    #
    # p_res = AbsResidualPlot(figure_name="absresidual_4z")
    # desc = "Generating plots"
    # for name in tqdm(r, desc=desc):
    #     p_res.plot(residuals[name], name)
    # p_res.ax.set_xlim(-100, 100)
    # p_res.ax.set_ylim(0.8, 6)
    # p_res.save()


if __name__ == '__main__':
    main()
