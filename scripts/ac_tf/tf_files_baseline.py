import numpy as np
from os.path import join, dirname, basename, splitext
from target_io import TargetIOEventReader as TIOReader
from target_io import T_SAMPLES_PER_WAVEFORM_BLOCK as N_BLOCKSAMPLES
from glob import glob
import re
import pandas as pd
from core import Plotter, plot_dir
from tqdm import tqdm
from IPython import embed


SAMPLES = 10

# CHEC-S
N_ROWS = 8
N_COLUMNS = 16
N_BLOCKS = N_ROWS * N_COLUMNS
N_CELLS = N_ROWS * N_COLUMNS * N_BLOCKSAMPLES
SKIP_SAMPLE = 0
SKIP_END_SAMPLE = 0
SKIP_EVENT = 2
SKIP_END_EVENT = 1


class Reader:
    def __init__(self, path):
        self.path = path

        self.reader = TIOReader(self.path, N_CELLS,
                                SKIP_SAMPLE, SKIP_END_SAMPLE,
                                SKIP_EVENT, SKIP_END_EVENT, True)

        self.is_r1 = self.reader.fR1
        if not self.is_r1:
            raise IOError("This script is only setup to read *_r1.tio files!")

        self.n_events = self.reader.fNEvents
        self.run_id = self.reader.fRunID
        self.n_pix = self.reader.fNPixels
        self.n_modules = self.reader.fNModules
        self.n_tmpix = self.n_pix // self.n_modules
        self.n_samples = self.reader.fNSamples
        self.n_cells = self.reader.fNCells
        self.time_ns = None
        self.time_sec = None

        self.max_blocksinwf = self.n_samples // N_BLOCKSAMPLES + 1
        self.samples = np.zeros((self.n_pix, self.n_samples), dtype=np.float32)
        self.first_cell_ids = np.zeros(self.n_pix, dtype=np.uint16)

        directory = dirname(path)
        filename = splitext(basename(path))[0]
        self.plot_directory = join(directory, filename)

    def get_event(self, iev):
        self.reader.GetR1Event(iev, self.samples, self.first_cell_ids)
        self.time_ns = self.reader.fCurrentTimeNs
        self.time_sec = self.reader.fCurrentTimeSec

    def event_generator(self):
        for iev in range(self.n_events):
            self.get_event(iev)
            yield iev


class Plot(Plotter):
    def plot(self, df, label=""):
        x = df['amplitude']
        y = df['mean']
        dy = df['stddev']

        self.ax.errorbar(x, y, yerr=dy, mew=1,
                         markersize=3, capsize=3, elinewidth=0.7, label=label)

    def finish(self):
        self.ax.axhline(0, linestyle='--', alpha=0.4)
        self.ax.set_xlabel("Input Amplitude (mV)")
        self.ax.set_ylabel("Baseline (Average and Stddev over all\n events "
                           "for first {} samples)".format(SAMPLES))
        self.ax.set_xscale("log", nonposy='clip')

    def zoom(self):
        self.ax.set_xlim(-100, 100)


def main():
    storage_path = join(plot_dir, "tf_files_baseline.h5")
    dfl = []
    dfgl = []
    event_count = 0
    pixel = 0

    path = '/Volumes/gct-jason/data_checs/tf/ac_tf_tmSN0074/Amplitude_*_r1.tio'
    pattern = 'Amplitude_(.+?)_r1.tio'
    file_list = glob(path)
    readers = [Reader(fp) for fp in file_list]

    desc1 = "Looping through files"

    for reader in tqdm(readers, desc=desc1):
        fp = reader.path
        amplitude = int(re.search(pattern, fp).group(1))
        n_events = reader.n_events
        n_pixels = reader.n_pix
        source = reader.event_generator()
        pedestal = np.zeros((n_events, n_pixels))
        time =
        desc2 = "Looping through events"
        for ev in tqdm(source, total=n_events, desc=desc2):
            waveforms = reader.samples
            pedestal[ev] = np.mean(waveforms[:, :SAMPLES], 1)

        mean = np.mean(pedestal, 0)
        stddev = np.std(pedestal, 0)
        d = dict(
            path=fp,
            amplitude=amplitude,
            mean=mean,
            stddev=stddev,
            pixel=np.arange(n_pixels)
        )
        df_f = pd.DataFrame(d)
        dfl.append(df_f)

        blocksize = n_events//25
        n_blocks = n_events//blocksize
        for blk in range(n_blocks):
            start = blk * blocksize
            end = (blk + 1) * blocksize
            ped_blk = pedestal[start:end]
            mean = np.mean(ped_blk, 0)
            stddev = np.std(ped_blk, 0)
            event_count += blocksize
            time =
            d = dict(
                type='point',
                path=fp,
                amplitude=amplitude,
                mean=mean,
                stddev=stddev,
                pixel=np.arange(n_pixels),
            )
            df_g = pd.DataFrame(d)
            dfgl.append(df_g)
            d = dict(
                type='point',
                path=fp,
                amplitude=amplitude,
                mean=mean,
                stddev=stddev,
                pixel=np.arange(n_pixels),
            )




    df = pd.concat(dfl)

    df = df.sort_values(['pixel', 'amplitude'])

    store = pd.HDFStore(storage_path)
    store['df'] = df

    store = pd.HDFStore(storage_path)
    df = store['df']

    p_single_pixel = Plot('baseline_single_pixel')
    df_pix = df.loc[df['pixel'] == pix]
    plot.plot(df_pix, pix)
    plot.save()

    p_single_asic = Plot('baseline_single_asic')
    for pix in range(16):
        df_pix = df.loc[df['pixel'] == pix]
        plot.plot(df_pix, pix)
    plot.add_legend("upper left")
    plot.save()


if __name__ == '__main__':
    main()
