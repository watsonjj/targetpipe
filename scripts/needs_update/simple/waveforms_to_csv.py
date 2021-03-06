import argparse
import numpy as np
from os.path import join, dirname, basename, splitext, exists
from os import makedirs
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

        self.max_blocksinwf = self.n_samples // N_BLOCKSAMPLES + 1
        self.samples = np.zeros((self.n_pix, self.n_samples), dtype=np.float32)
        self.first_cell_ids = np.zeros(self.n_pix, dtype=np.uint16)

        directory = dirname(path)
        self.filename = splitext(basename(path))[0]
        self.plot_directory = join(directory, self.filename)

    def get_event(self, iev):
        self.reader.GetR1Event(iev, self.samples, self.first_cell_ids)

    def event_generator(self):
        for iev in range(self.n_events):
            self.get_event(iev)
            yield iev


def main():
    description = 'Check for missing pedestal values'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the TIO r1 run file')
    args = parser.parse_args()

    reader = Reader(args.input_path)
    source = reader.event_generator()

    n_events = reader.n_events
    n_samples = reader.n_samples
    waveforms = np.zeros((n_events, n_samples))
    mask = np.zeros((n_events, n_samples), dtype=np.bool)

    pixel = 35

    for ev in source:
        # Skip first row due to problem in pedestal subtraction
        bp, r, c = get_bp_r_c(reader.first_cell_ids[0])
        if r == 0:
            mask[ev] = True
            continue

        waveforms_ev = reader.samples

        waveforms_pix = waveforms_ev[pixel]
        waveforms[ev] = waveforms_pix

    waveforms = np.ma.masked_array(waveforms, mask=mask).compressed()
    waveforms = waveforms.reshape((waveforms.size // n_samples, n_samples))

    output_dir = reader.plot_directory
    if not exists(output_dir):
        print("Creating directory: {}".format(output_dir))
        makedirs(output_dir)

    run_name = reader.filename
    output_path = join(output_dir, "{}_wf_ch{}.csv".format(run_name, pixel))
    np.savetxt(output_path, waveforms, delimiter=",", fmt='%.6f')


if __name__ == '__main__':
    main()
