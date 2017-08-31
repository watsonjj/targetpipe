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
from scipy.stats.distributions import poisson
import iminuit
from tqdm import tqdm
from fit_algorithms import sipm_spe_fit, pedestal_signal, pe_signal
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


class Cleaner:
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


class Fitter:
    def __init__(self):
        self.coeff_list = []
        self.initial = dict()
        self.limits = dict()
        self.subfit_labels = []

        self.hist = None
        self.edges = None
        self.between = None
        self.fit_x = None
        self.fit = None
        self.coeff = None

        self.nbins = 60
        self.range = [-30, 100]

        self.add_parameter("norm", None, 0, 1000000)
        self.add_parameter("eped", -30, -50, 50)
        self.add_parameter("eped_sigma", 10, 0, 50)
        self.add_parameter("spe", 30, 0, 50)
        self.add_parameter("spe_sigma", 10, 0, 50)
        self.add_parameter("lambda_", 1, 0, 5)
        self.add_parameter("opct", 0.65, 0.1, 0.9)
        self.add_parameter("pap", 0.1, 0.1, 0.9)
        self.add_parameter("dap", 0.3, 0.1, 0.9)

        self.k = np.arange(1, 11)
        self.subfit_labels = ['pedestal']
        for pe in self.k:
            self.subfit_labels.append('{}pe'.format(pe))

    def add_parameter(self, name, inital, lower, upper):
        self.coeff_list.append(name)
        self.initial[name] = inital
        self.limits["limit_" + name] = (lower, upper)

    def apply(self, spectrum):
        hist, edges = self.get_histogram(spectrum)
        between = (edges[1:] + edges[:-1]) / 2
        fit_x = np.linspace(edges[0], edges[-1], edges.size*10)

        self.hist = hist
        self.edges = edges
        self.between = between
        self.fit_x = fit_x

        try:
            fit, coeff = self._fit(hist, edges, between, fit_x)
        except RuntimeError:
            self.fit = [0]*len(fit_x)
            self.coeff = {}
            return False

        self.fit = fit
        self.coeff = coeff
        return True

    def get_histogram(self, spectrum):
        range_ = self.range[:]
        hist, edges = np.histogram(spectrum, bins=self.nbins, range=range_)
        return hist, edges

    @property
    def subfits(self):
        return self._get_subfits()

    def _fit(self, hist, edges, between, fit_x):
        p0 = self.initial.copy()
        limits = self.limits.copy()
        if p0['norm'] is None:
            p0['norm'] = np.sum(np.diff(edges) * hist)
        fit, coeff = self.iminuit_fit(between, hist, p0, fit_x, limits)
        return fit, coeff

    @staticmethod
    def iminuit_fit(x, y, p0, fit_x=None, limits=None):
        fit = sipm_spe_fit
        if fit_x is None:
            fit_x = x
        if limits is None:
            limits = {}

        def minimizehist(norm, eped, eped_sigma, spe, spe_sigma, lambda_, opct, pap, dap):
            p = fit(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_, opct, pap, dap)
            like = -2 * poisson.logpmf(y, p)
            return np.nansum(like)

        m0 = iminuit.Minuit(minimizehist, **p0, **limits,
                            print_level=0, pedantic=False, throw_nan=False)
        m0.migrad()

        result = fit(fit_x, **m0.values)
        return result, m0.values

    def _get_subfits(self):
        if self.coeff:
            def pedestal_kw(x, norm, eped, eped_sigma, lambda_, **kw):
                return pedestal_signal(x, norm, eped, eped_sigma, lambda_)

            def pe_kw(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_, opct, pap, dap, **kw):
                return pe_signal(self.k[:, None], x[None, :], norm, eped, eped_sigma, spe, spe_sigma, lambda_, opct, pap, dap)

            pedestal = pedestal_kw(self.fit_x, **self.coeff)
            pe_fits = pe_kw(self.fit_x, **self.coeff)
            subfits = dict(pedestal=pedestal)
            for i, pe_i in enumerate(self.k):
                subfits['{}pe'.format(pe_i)] = pe_fits[i]
        else:
            subfits = dict(pedestal=np.zeros(self.fit_x.shape))
            for i, pe_i in enumerate(self.k):
                subfits['{}pe'.format(pe_i)] = np.zeros(self.fit_x.shape)

        return subfits


def main():
    description = 'Check for missing pedestal values'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the TIO r1 run file')
    parser.add_argument('-p', '--pixel', dest='pixel', action='store',
                        required=True, help='pixel to investigate', type=int)
    args = parser.parse_args()

    poi = args.pixel

    reader = Reader(args.input_path)
    source = reader.event_generator()
    n_events = reader.n_events

    cleaner = Cleaner()
    fitter = Fitter()

    connected = list(range(reader.n_pix))
    ignore_pixels = [25, 26, 18, 19, 13, 14, 5, 6, 49, 37]
    connected = [i for i in connected if i not in ignore_pixels]

    df_list = []

    desc = "Looping over events"
    for ev in tqdm(source, total=n_events, desc=desc):
        bp, r, c = get_bp_r_c(reader.first_cell_ids)

        waveforms = reader.samples
        cleaned = cleaner.clean(waveforms)
        avgwf = np.mean(cleaned[connected], 0)
        avgwf_t0 = np.argmax(avgwf)

        height_at_t0 = cleaned[:, avgwf_t0]

        df_list.append(dict(
            event=ev,
            row=r[poi],
            avgwf=np.copy(avgwf),
            wf=np.copy(waveforms[poi]),
            cleanedwf=cleaned[poi],
            t0=avgwf_t0,
            height_at_t0=height_at_t0[poi]
        ))

    df = pd.DataFrame(df_list)
    df = df.loc[df['row'] != 0]

    output_dir = join(reader.plot_directory, "checs_fit", "p{}".format(poi))
    if not exists(output_dir):
        print("Creating directory: {}".format(output_dir))
        makedirs(output_dir)

    # embed()

    f_heightatt0_fit = plt.figure(figsize=(14, 10))
    ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    axt = plt.subplot2grid((1, 3), (0, 2))
    range_ = [-90, 180]#[-90, 400]
    bins = 110
    increment = (range_[1] - range_[0]) / bins
    v = df['height_at_t0'].values
    fitter.range = range_
    fitter.nbins = bins
    fitter.apply(v)
    h = fitter.hist
    e = fitter.edges
    b = fitter.between
    fitx = fitter.fit_x
    fit = fitter.fit
    coeff = fitter.coeff
    coeff_l = fitter.coeff_list
    ax.hist(b, bins=e, weights=h, histtype='step')
    ax.plot(fitx, fit, label="Fit")
    for sf in fitter.subfit_labels:
        arr = fitter.subfits[sf]
        ax.plot(fitx, arr, label=sf)
    ax.set_title("Height At T0 Fit (Pixel {})".format(poi))
    ax.set_xlabel("Height At T0")
    ax.set_ylabel("N")
    ax.xaxis.set_minor_locator(MultipleLocator(increment * 2))
    ax.xaxis.set_major_locator(MultipleLocator(increment * 10))
    ax.xaxis.grid(b=True, which='minor', alpha=0.5)
    ax.xaxis.grid(b=True, which='major', alpha=0.8)
    ax.legend(loc=1)
    axt.axis('off')
    table_data = [['%.3f' % coeff[i]] for i in coeff_l]
    table_row = coeff_l
    table = axt.table(cellText=table_data, rowLabels=table_row, loc='center')
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    output_path = join(output_dir, "heightatt0_fit")
    f_heightatt0_fit.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))

if __name__ == '__main__':
    main()
