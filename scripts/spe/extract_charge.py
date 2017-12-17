import argparse
from os.path import join, dirname, basename, splitext, exists
from os import makedirs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from target_io import TargetIOEventReader as TIOReader
from target_io import T_SAMPLES_PER_WAVEFORM_BLOCK as N_BLOCKSAMPLES
from scipy import interpolate
from scipy.ndimage import correlate1d
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
        filename = splitext(basename(path))[0]
        self.plot_directory = join(directory, filename)

    def get_event(self, iev):
        self.reader.GetR1Event(iev, self.samples, self.first_cell_ids)

    def event_generator(self):
        for iev in range(self.n_events):
            self.get_event(iev)
            yield iev


class CrossCorrelation:
    def __init__(self):
        file = np.loadtxt("pulse_data.txt", delimiter=', ')
        refx = file[:, 0]
        refy = file[:, 1] - file[:, 1][0]
        f = interpolate.interp1d(refx, refy, kind=3)
        x = np.linspace(0, 77e-9, 76)
        y = f(x)
        self.reference_pulse = y

    def clean(self, waveforms):
        cleaned = correlate1d(waveforms, self.reference_pulse)
        return cleaned


def main():
    description = 'Extract charge from a SPE tio file'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the TIO r1 run file')
    parser.add_argument('-p', '--pixel', dest='pixel', action='store',
                        required=True, help='pixel to investigate', type=int)
    parser.add_argument('-e', '--event', dest='event', action='store',
                        required=True, help='event to plot waveform', type=int)
    args = parser.parse_args()

    poi = args.pixel
    eoi = args.event

    reader = Reader(args.input_path)
    n_events = reader.n_events
    source = reader.event_generator()

    cccleaner = CrossCorrelation()

    df_list = []
    raw_wf = None
    cleaned_wf = None
    avg_wf = None
    t0_eoi = None

    desc = "Looping over events"
    for ev in tqdm(source, total=n_events, desc=desc):
        bp, r, c = get_bp_r_c(reader.first_cell_ids)

        waveforms = reader.samples

        # Cross Correlation Method
        cleaned = cccleaner.clean(waveforms)
        avg = np.mean(cleaned, 0)
        avg_t0 = np.argmax(avg)
        charge = cleaned[:, avg_t0]

        # # Integration Window Method
        # cleaned = waveforms
        # shift = 4
        # window = 8
        # avg = np.mean(cleaned, 0)
        # avg_t0 = np.argmax(avg)
        # start = avg_t0 - shift
        # end = start + window
        # charge = cleaned[:, start:end].sum(1)

        if ev == eoi:
            raw_wf = np.copy(waveforms)
            cleaned_wf = np.copy(cleaned)
            avg_wf = np.copy(avg)
            t0_eoi = avg_t0

        df_list.append(dict(
            event=ev,
            row=r[poi],
            t0=avg_t0,
            charge=np.copy(charge)
        ))

    df = pd.DataFrame(df_list)
    # df = df.loc[df['row'] != 0]
    charge = np.vstack(df['charge'])

    output_dir = join(reader.plot_directory, "spe_extract_charge")
    if not exists(output_dir):
        print("Creating directory: {}".format(output_dir))
        makedirs(output_dir)

    output_path = join(output_dir, "charge.npy")
    np.save(output_path, charge)
    print("Numpy array saved to: {}".format(output_path))

    f_hist = plt.figure(figsize=(14, 10))
    ax = f_hist.add_subplot(1, 1, 1)
    range_ = [-50, 250]
    bins = 100
    increment = (range_[1] - range_[0]) / bins
    ax.hist(charge[:, poi], bins=bins, range=range_, histtype='step')
    ax.set_title("Charge (pixel = {})".format(poi))
    ax.set_xlabel("Charge")
    ax.set_ylabel("N")
    ax.xaxis.set_minor_locator(MultipleLocator(6))
    ax.xaxis.set_major_locator(MultipleLocator(increment * 10))
    ax.xaxis.grid(b=True, which='minor', alpha=0.5)
    ax.xaxis.grid(b=True, which='major', alpha=0.8)
    output_path = join(output_dir, "charge_hist_p{}.pdf".format(poi))
    f_hist.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))

    f_raw = plt.figure(figsize=(14, 10))
    ax = f_raw.add_subplot(1, 1, 1)
    ax.plot(raw_wf.T)
    ax.axvline(t0_eoi, c='red')
    ax.set_title("Raw Wfs (event = {})".format(eoi))
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.xaxis.set_major_locator(MultipleLocator(16))
    output_path = join(output_dir, "wf_ev{}_raw.pdf".format(eoi))
    f_raw.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))

    f_cleaned = plt.figure(figsize=(14, 10))
    ax = f_cleaned.add_subplot(1, 1, 1)
    ax.plot(cleaned_wf.T)
    ax.axvline(t0_eoi, c='red')
    ax.set_title("Cleaned Wfs (event = {})".format(eoi))
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.xaxis.set_major_locator(MultipleLocator(16))
    output_path = join(output_dir, "wf_ev{}_cleaned.pdf".format(eoi))
    f_cleaned.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))

    f_avg = plt.figure(figsize=(14, 10))
    ax = f_avg.add_subplot(1, 1, 1)
    ax.plot(avg_wf)
    ax.axvline(t0_eoi, c='red')
    ax.set_title("Average Wf (event = {})".format(eoi))
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.xaxis.set_major_locator(MultipleLocator(16))
    output_path = join(output_dir, "wf_ev{}_avg.pdf".format(eoi))
    f_avg.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))

    f_raw_poi = plt.figure(figsize=(14, 10))
    ax = f_raw_poi.add_subplot(1, 1, 1)
    ax.plot(raw_wf[poi])
    ax.axvline(t0_eoi, c='red')
    ax.set_title("Raw Wf (event = {}, pixel = {})".format(eoi, poi))
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.xaxis.set_major_locator(MultipleLocator(16))
    output_path = join(output_dir, "wf_ev{}_p{}_raw.pdf".format(eoi, poi))
    f_raw_poi.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))

    f_cleaned_poi = plt.figure(figsize=(14, 10))
    ax = f_cleaned_poi.add_subplot(1, 1, 1)
    ax.plot(cleaned_wf[poi])
    ax.axvline(t0_eoi, c='red')
    ax.set_title("Cleaned Wf (event = {}, pixel = {})".format(eoi, poi))
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.xaxis.set_major_locator(MultipleLocator(16))
    output_path = join(output_dir, "wf_ev{}_p{}_cleaned.pdf".format(eoi, poi))
    f_cleaned_poi.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))


if __name__ == '__main__':
    main()
