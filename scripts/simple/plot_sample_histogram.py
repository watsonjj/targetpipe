import argparse
from os.path import exists, join, dirname, basename, splitext
from os import makedirs
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from target_io import TargetIOEventReader as TIOReader
from target_io import T_SAMPLES_PER_WAVEFORM_BLOCK as N_BLOCKSAMPLES
from target_calib import RunningStats
from IPython import embed


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
                                SKIP_EVENT, SKIP_END_EVENT)

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

        self.max_blocksinwf = self.n_samples // N_BLOCKSAMPLES + 1
        self.samples = np.zeros((self.n_pix, self.n_samples), dtype=np.float32)
        self.first_cell_ids = np.zeros(self.n_pix, dtype=np.uint16)

        directory = dirname(path)
        filename = splitext(basename(path))[0]
        self.plot_directory = join(directory, filename)

    def get_event(self, iev):
        self.reader.GetR1Event(iev, self.samples, self.first_cell_ids)

    def event_generator(self):
        for iev in range(self.n_events):
            self.get_event(iev)
            yield iev


class HistPlotter:
    name = 'HistPlotter'

    def __init__(self, bins, range_):
        self.fig = plt.figure(figsize=(8, 4))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.hist, self.edges = np.histogram(np.nan, bins=bins, range=range_)
        self.between = (self.edges[1:] + self.edges[:-1]) / 2
        self.rs = RunningStats()

    def add(self, data):
        data = data.ravel()

        self.rs.PushArray(data)

        self.hist += np.histogram(data, bins=self.edges)[0]

    def plot(self):
        label = "(Mean = {:.2f}, Stddev = {:.2f}, N = {:.2e})".format(
            self.rs.Mean(),
            self.rs.StandardDeviation(),
            self.rs.NumDataValues()
        )
        self.ax.hist(self.between, bins=self.edges, weights=self.hist,
                     label=label, alpha=0.7)

        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("N")
        # self.ax.xaxis.set_major_locator(MultipleLocator(16))

    def save(self, output_path=None):
        output_dir = dirname(output_path)

        self.ax.legend(loc="upper left", fontsize=5)
        if not exists(output_dir):
            print("Creating directory: {}".format(output_dir))
            makedirs(output_dir)

        self.fig.savefig(output_path, bbox_inches='tight')
        print("Figure saved to: {}".format(output_path))


def main():
    description = 'Simple Waveform Reader'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the TIO r1 run file')
    args = parser.parse_args()

    reader = Reader(args.input_path)
    n_events = reader.n_events
    source = reader.event_generator()

    hist = HistPlotter(100, (-10, 10))

    desc = "Looping through events"
    for _ in tqdm(source, total=n_events, desc=desc):
        waveforms = reader.samples
        hist.add(waveforms)

    hist.plot()

    output_dir = join(
        reader.plot_directory,
        "plot_sample_histogram",
        "hist.pdf"
    )
    hist.save(output_dir)


if __name__ == '__main__':
    main()
