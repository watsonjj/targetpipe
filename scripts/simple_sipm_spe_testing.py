import argparse
import numpy as np
from scipy.signal import general_gaussian
from matplotlib.ticker import MultipleLocator
from matplotlib import pyplot as plt
from os.path import exists, join, dirname, basename, splitext
from os import makedirs
from target_io import TargetIOEventReader as TIOReader
from target_io import T_SAMPLES_PER_WAVEFORM_BLOCK as N_BLOCKSAMPLES
from tqdm import tqdm


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


def main():
    description = 'Check for missing pedestal values'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the TIO r1 run file')
    args = parser.parse_args()

    reader = Reader(args.input_path)
    source = reader.event_generator()

    # Set pixel of interest
    connected_pixels = list(range(reader.n_pix))
    ignore_pixels = [25, 26, 18, 19, 13, 14, 5, 6, 49, 37]
    connected_pixels = [i for i in connected_pixels if i not in ignore_pixels]
    pix = 42
    pix_index = connected_pixels.index(pix)

    # Define integration window parameters
    shift = 4
    width = 8
    match_t0 = 60

    n_pixels = len(connected_pixels)
    n_samples = reader.n_samples

    # Define window
    window_start = match_t0 - shift
    if window_start < 0:
        window_start = 0
    window_end = window_start + width
    if window_end > n_samples - 1:
        window_end = n_samples - 1

    # Create empty arrays to be filled
    n_events = reader.n_events
    mask = np.zeros(n_events, dtype=np.bool)
    area = np.zeros(n_events)
    height_t0 = np.zeros(n_events)
    height_peak = np.zeros(n_events)
    peakpos = np.zeros(n_events)
    t0 = np.zeros(n_events)
    t0_chk = np.zeros(n_events)
    waveform_pix = np.zeros((n_events, n_samples))
    waveform_avg = np.zeros((n_events, n_samples))

    cut_row = 0

    desc = "Looping through events"
    for ev in tqdm(source, total=n_events, desc=desc):
        # Skip first row due to problem in pedestal subtraction
        bp, r, c = get_bp_r_c(reader.first_cell_ids[0])
        if r == 0:
            cut_row += 1
            mask[ev] = True
            continue

        r1 = reader.samples[connected_pixels]

        # Waveform cleaning
        kernel = general_gaussian(5, p=1.0, sig=1)
        smooth_flat = np.convolve(r1.ravel(), kernel, "same")
        smoothed = np.reshape(smooth_flat, r1.shape)
        samples_std = np.std(r1, axis=1)
        smooth_baseline_std = np.std(smoothed, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            smoothed *= (samples_std / smooth_baseline_std)[:, None]
            smoothed[~np.isfinite(smoothed)] = 0
        r1 = smoothed

        # Get average pulse of event
        avg = r1.mean(0)

        # Get t0 of average pulse for event
        t0_ev = avg.argmax()

        # Shift waveform to match t0 between events
        t0_ev_shift = t0_ev - 60
        r1_shift = np.zeros((n_pixels, n_samples))
        if t0_ev_shift >= 0:
            r1_shift[:, :r1[:, t0_ev_shift:].shape[1]] = r1[:, t0_ev_shift:]
        else:
            r1_shift[:, r1[:, t0_ev_shift:].shape[1]:] = r1[:, :t0_ev_shift]

        # Check t0 matching
        avg_chk = r1_shift.mean(0)
        t0_chk_ev = avg_chk.argmax()

        # Extract area and height for pixel
        r1_pix = r1_shift[pix_index]
        area_ev = r1_pix[window_start:window_end].sum()
        height_t0_ev = r1_pix[match_t0]
        height_peak_ev = r1_pix[window_start:window_end].max()
        peakpos_ev = r1_pix[window_start:window_end].argmax() + window_start

        area[ev] = area_ev
        height_t0[ev] = height_t0_ev
        height_peak[ev] = height_peak_ev
        peakpos[ev] = peakpos_ev
        t0[ev] = t0_ev
        t0_chk[ev] = t0_chk_ev
        waveform_pix[ev] = r1_pix
        waveform_avg[ev] = avg

    def remove_events(array):
        array = np.ma.masked_array(array, mask=mask).compressed()
        return array

    def remove_events_samples(array):
        mask_samples = np.zeros((n_events, n_samples), dtype=np.bool)
        mask_samples = np.ma.mask_or(mask_samples, mask[:, None])
        array = np.ma.masked_array(array, mask=mask_samples).compressed()
        array = array.reshape((array.size // n_samples, n_samples))
        return array

    area = remove_events(area)
    height_t0 = remove_events(height_t0)
    height_peak = remove_events(height_peak)
    peakpos = remove_events(peakpos)
    t0 = remove_events(t0)
    t0_chk = remove_events(t0_chk)
    waveform_pix = remove_events_samples(waveform_pix)
    waveform_avg = remove_events_samples(waveform_avg)

    print("Number of events removed by row cut = {}".format(cut_row))
    print("Number of events after cuts = {}".format(area.size))

    output_dir = join(reader.plot_directory, "sipm_spe_testing")
    output_dir = join(output_dir, "ch{}".format(pix))
    if not exists(output_dir):
        print("Creating directory: {}".format(output_dir))
        makedirs(output_dir)

    output_path = join(output_dir, "area.npy")
    np.save(output_path, area)
    print("Numpy array saved to: {}".format(output_path))

    f_area_spectrum = plt.figure(figsize=(14, 10))
    ax = f_area_spectrum.add_subplot(1, 1, 1)
    range_ = [-80, 340]
    bins = 140
    increment = (range_[1] - range_[0]) / bins
    ax.hist(area, bins=bins, range=range_)
    ax.set_title("Area Spectrum (Channel {})".format(pix))
    ax.set_xlabel("Area")
    ax.set_ylabel("N")
    ax.xaxis.set_minor_locator(MultipleLocator(increment * 2))
    ax.xaxis.set_major_locator(MultipleLocator(increment * 10))
    ax.xaxis.grid(b=True, which='minor', alpha=0.5)
    ax.xaxis.grid(b=True, which='major', alpha=0.8)
    output_path = join(output_dir, "area_spectrum")
    f_area_spectrum.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))

    f_height_t0_spectrum = plt.figure(figsize=(14, 10))
    ax = f_height_t0_spectrum.add_subplot(1, 1, 1)
    range_ = [-10, 45]
    bins = 110
    increment = (range_[1] - range_[0]) / bins
    ax.hist(height_t0, bins=bins, range=range_)
    ax.set_title("Height t0 Spectrum (Channel {})".format(pix))
    ax.set_xlabel("Height (@t0)")
    ax.set_ylabel("N")
    ax.xaxis.set_minor_locator(MultipleLocator(increment * 2))
    ax.xaxis.set_major_locator(MultipleLocator(increment * 10))
    ax.xaxis.grid(b=True, which='minor', alpha=0.5)
    ax.xaxis.grid(b=True, which='major', alpha=0.8)
    ax.xaxis.set_minor_locator(MultipleLocator(increment))
    output_path = join(output_dir, "height_t0_spectrum")
    f_height_t0_spectrum.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))

    f_height_peak_spectrum = plt.figure(figsize=(14, 10))
    ax = f_height_peak_spectrum.add_subplot(1, 1, 1)
    range_ = [-5, 50]
    bins = 110
    increment = (range_[1] - range_[0]) / bins
    ax.hist(height_peak, bins=bins, range=range_)
    ax.set_title("Height Peak Spectrum (Channel {})".format(pix))
    ax.set_xlabel("Height (@Peak)")
    ax.set_ylabel("N")
    ax.xaxis.set_minor_locator(MultipleLocator(increment * 2))
    ax.xaxis.set_major_locator(MultipleLocator(increment * 10))
    ax.xaxis.grid(b=True, which='minor', alpha=0.5)
    ax.xaxis.grid(b=True, which='major', alpha=0.8)
    ax.xaxis.set_minor_locator(MultipleLocator(increment))
    output_path = join(output_dir, "height_peak_spectrum")
    f_height_peak_spectrum.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))

    f_peakpos_hist = plt.figure(figsize=(14, 10))
    ax = f_peakpos_hist.add_subplot(1, 1, 1)
    min_ = peakpos.min()
    max_ = peakpos.max()
    nbins = int(max_ - min_)
    ax.hist(peakpos, bins=nbins, range=[min_, max_])
    ax.set_title("peakpos Distribution (Channel {})".format(pix))
    ax.set_xlabel("peakpos")
    ax.set_ylabel("N")
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    output_path = join(output_dir, "peakpos_hist")
    f_peakpos_hist.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))

    f_t0_hist = plt.figure(figsize=(14, 10))
    ax = f_t0_hist.add_subplot(1, 1, 1)
    min_ = t0.min()
    max_ = t0.max()
    nbins = int(max_ - min_)
    ax.hist(t0, bins=nbins, range=[min_, max_])
    ax.set_title("t0 Distribution")
    ax.set_xlabel("t0")
    ax.set_ylabel("N")
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    output_path = join(output_dir, "t0_hist")
    f_t0_hist.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))

    f_t0_chk_hist = plt.figure(figsize=(14, 10))
    ax = f_t0_chk_hist.add_subplot(1, 1, 1)
    min_ = t0_chk.min()
    max_ = t0_chk.max()
    nbins = int(max_ - min_ + 1)
    ax.hist(t0_chk, bins=nbins, range=[min_, max_])
    ax.set_title("t0 (matched) Distribution")
    ax.set_xlabel("t0")
    ax.set_ylabel("N")
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    output_path = join(output_dir, "t0_chk_hist")
    f_t0_chk_hist.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))

    f_waveforms_pix = plt.figure(figsize=(14, 10))
    ax = f_waveforms_pix.add_subplot(1, 1, 1)
    ax.plot(np.rollaxis(waveform_pix, 1))
    plt.axvline(x=60, color="green")
    plt.axvline(x=window_start, color="red")
    plt.axvline(x=window_end, color="red")
    ax.set_title("Waveforms (Channel {})".format(pix))
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude (ADC pedsub)")
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    output_path = join(output_dir, "waveforms_pix")
    f_waveforms_pix.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))

    f_waveforms_pix_hist = plt.figure(figsize=(14, 10))
    ax = f_waveforms_pix_hist.add_subplot(1, 1, 1)
    x = np.indices(waveform_pix.shape)[1].ravel()
    y = waveform_pix.ravel()
    hb = ax.hexbin(x, y, bins='log')
    cb = f_waveforms_pix_hist.colorbar(hb, ax=ax)
    cb.set_label('Counts (log)')
    plt.axvline(x=60, color="green")
    plt.axvline(x=window_start, color="red")
    plt.axvline(x=window_end, color="red")
    ax.set_title("Waveforms (Channel {})".format(pix))
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude (ADC pedsub)")
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    output_path = join(output_dir, "waveforms_pix_hist")
    f_waveforms_pix_hist.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))

    f_waveforms_pix_hist_spe = plt.figure(figsize=(14, 10))
    ax = f_waveforms_pix_hist_spe.add_subplot(1, 1, 1)
    x = np.indices(waveform_pix.shape)[1].ravel()
    y = waveform_pix.ravel()
    hb = ax.hexbin(x, y, bins='log')
    cb = f_waveforms_pix_hist_spe.colorbar(hb, ax=ax)
    cb.set_label('Counts (log)')
    plt.axvline(x=60, color="green")
    plt.axvline(x=window_start, color="red")
    plt.axvline(x=window_end, color="red")
    ax.set_title("Waveforms (Channel {})".format(pix))
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude (ADC pedsub)")
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    output_path = join(output_dir, "waveforms_pix_hist_spe")
    f_waveforms_pix_hist_spe.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))

    f_waveforms_avg = plt.figure(figsize=(14, 10))
    ax = f_waveforms_avg.add_subplot(1, 1, 1)
    ax.plot(np.rollaxis(waveform_avg, 1))
    ax.set_title("Average Waveforms")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude (ADC pedsub)")
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    output_path = join(output_dir, "waveforms_avg")
    f_waveforms_avg.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))

if __name__ == '__main__':
    main()