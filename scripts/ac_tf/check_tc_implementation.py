import sys
from os import environ
from os.path import join
from glob import glob
import re
import numpy as np
from matplotlib import pyplot as plt
import target_calib
from core import input_path, Plotter, plot_dir
from tf import TFStorageCell, TFTargetCalib
import pandas as pd
from copy import copy
from IPython import embed
from tqdm import tqdm
import pickle


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


class TFPlot(Plotter):
    def plot(self, x, y, label):
        self.ax.plot(x, y, mew=0.5, label=label)
        # self.ax.plot(x, y, mew=0.5, label=label)

        self.ax.set_xlabel("ADC")
        self.ax.set_ylabel("Amplitude (mV)")
        self.ax.legend(loc="upper right")


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
    pickle_path = join(plot_dir, "check_tc_implementation.p")

    store = pd.HDFStore(input_path)
    df = store['df']

    tc_class = TFTargetCalib()
    # tc_class.create(df)

    sc_class = TFStorageCell()
    # sc_class.create(df)

    # tc_vped, tc_tfinput = tc_class.get_tfinput(df)
    # sc_vped, sc_tfinput = sc_class.get_tfinput(df)
    #
    # assert np.allclose(tc_tfinput, sc_tfinput)
    #
    # tc_adcx, tc_tf = tc_class._load_tf()
    # sc_adcx, sc_tf = sc_class._load_tf()
    #
    # tc_tfplot = TFPlot("tf_tc")
    # tc_tfplot.plot(tc_adcx, tc_tf.T, "")
    # tc_tfplot.save()
    #
    # sc_tfplot = TFPlot("tf_sc")
    # sc_tfplot.plot(sc_adcx, sc_tf.T, "")
    # sc_tfplot.save()

    tc_df = copy(df)
    sc_df = copy(df)

    tc_df = tc_class.calibrate(tc_df)
    sc_df = sc_class.calibrate(sc_df)

    tqdm.pandas(desc="Obtaining residuals tc")
    tc_df = tc_df.groupby('vped').progress_apply(rmse_df).reset_index()
    tc_dfres = tc_df.loc[:, ['vped', 'rmse', 'rmseabs']].groupby('vped').first().reset_index()

    tqdm.pandas(desc="Obtaining residuals sc")
    sc_df = sc_df.groupby('vped').progress_apply(rmse_df).reset_index()
    sc_dfres = sc_df.loc[:, ['vped', 'rmse', 'rmseabs']].groupby('vped').first().reset_index()

    # embed()

    # pickle.dump([tc_dfres, sc_dfres], open(pickle_path, "wb"))

    # tc_df, sc_df = pickle.load(open(pickle_path, "rb"))

    resplot = ResidualPlot("res_comparison_tc")
    resplot.plot(tc_dfres, "tc")
    resplot.plot(sc_dfres, "sc")
    resplot.save()
    resplot = ResidualPlot("res_comparison_tc_z")
    resplot.plot(tc_dfres, "tc")
    resplot.plot(sc_dfres, "sc")
    resplot.zoom()
    resplot.save()


if __name__ == '__main__':
    main()
