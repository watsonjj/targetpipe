import argparse
import numpy as np
from math import ceil
from os.path import join, dirname, basename, splitext
from target_io import TargetIOEventReader as TIOReader
from target_io import T_SAMPLES_PER_WAVEFORM_BLOCK as N_BLOCKSAMPLES


# CHEC-S
N_ROWS = 8
N_COLUMNS = 16
N_BLOCKS = N_ROWS * N_COLUMNS
N_CELLS = N_ROWS * N_COLUMNS * N_BLOCKSAMPLES
SKIP_SAMPLE = 0
SKIP_END_SAMPLE = 0
SKIP_EVENT = 2
SKIP_END_EVENT = 1


def get_bp_r_c(cells):
    blockphase = cells % N_BLOCKSAMPLES
    row = (cells // N_BLOCKSAMPLES) % 8
    column = (cells // N_BLOCKSAMPLES) // 8
    return blockphase, row, column


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
        self.bunch_size = 100

        self.max_blocksinwf = self.n_samples // N_BLOCKSAMPLES + 1
        self.samples = np.zeros((self.bunch_size, self.n_pix, self.n_samples), dtype=np.float32)
        self.first_cell_ids = np.zeros((self.bunch_size, self.n_pix), dtype=np.uint16)

        directory = dirname(path)
        filename = splitext(basename(path))[0]
        self.plot_directory = join(directory, filename)

    def get_event_bunch(self, iev):
        self.reader.GetR1Events(iev, self.samples, self.first_cell_ids)

    def event_generator(self):
        for ibunch in range(ceil(self.n_events/self.bunch_size)):
            iev = ibunch * self.bunch_size
            self.get_event_bunch(iev)
            yield ibunch


def main():
    description = 'Check for missing pedestal values'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the TIO r1 run file')
    args = parser.parse_args()

    reader = Reader(args.input_path)
    source = reader.event_generator()

    for ev_bunch in source:
        waveforms = reader.samples


if __name__ == '__main__':
    main()
