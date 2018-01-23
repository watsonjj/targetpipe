import argparse
import numpy as np
from os.path import join, dirname, basename, splitext
from target_io import TargetIOEventReader as TIOReader
from target_io import T_SAMPLES_PER_WAVEFORM_BLOCK as N_BLOCKSAMPLES
from core import pix, pedestal_path
from target_calib import CalculateRowColumnBlockPhase, GetCellIDArray
import pandas as pd
from tqdm import tqdm
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


def main():
    input_path = "/Volumes/gct-jason/data_checs/tf/ac_tf_tmSN0074/Pedestals_r1.tio"

    reader = Reader(input_path)
    source = reader.event_generator()
    nevents = reader.n_events

    sample_i = np.arange(reader.n_samples, dtype=np.uint16)

    df_list = []

    desc = "Looping through events"
    for ev in tqdm(source, total=nevents, desc=desc):
        waveforms = reader.samples[pix]
        fci = np.asscalar(reader.first_cell_ids[pix])
        bp, r, c = CalculateRowColumnBlockPhase(fci)
        fb = c * 8 + r
        cell = GetCellIDArray(fci, sample_i)
        for w, cc, s in zip(waveforms, cell, sample_i):
            df_list.append(dict(event=ev, pixel=pix, row=r, column=c,
                                blockphase=bp, fci=fci, fb=fb, cell=cc,
                                vped=0, sample=s, adc=w))

    df = pd.DataFrame(df_list)
    store = pd.HDFStore(pedestal_path)
    store['df'] = df


if __name__ == '__main__':
    main()
