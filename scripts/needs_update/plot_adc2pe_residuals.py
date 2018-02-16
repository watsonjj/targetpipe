from traitlets import Dict, List, Unicode
from ctapipe.core import Tool
from targetpipe.io.pixels import Dead
import numpy as np
from os.path import join, exists, dirname
from os import makedirs
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats, integrate


class ADC2PEResidualsPlotter(Tool):
    name = "ADC2PEResidualsPlotter"
    description = "Plot the residuals from the adc2pe calibration"

    input_path = Unicode("", help="Path to the adc2pe_residuals numpy "
                                  "file").tag(config=True)

    aliases = Dict(dict(i='ADC2PEResidualsPlotter.input_path',
                        ))
    classes = List([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dead = None

        self.output_dir = None

        self.spe = None
        self.spe_sigma = None
        self.hist = None
        self.edges = None
        self.between = None

        self.fig_spectrum_all = None
        self.fig_spectrum_tm_list = None
        self.fig_combgaus = None
        self.fig_kde = None
        self.fig_hist = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"

        self.dead = Dead()

        file = np.load(self.input_path)
        self.spe = file['spe']
        self.spe_sigma = file['spe_sigma']
        self.hist = file['hist']
        self.edges = file['edges']
        self.between = file['between']

        self.output_dir = join(dirname(self.input_path),
                               "plot_adc2pe_residuals")
        if not exists(self.output_dir):
            self.log.info("Creating directory: {}".format(self.output_dir))
            makedirs(self.output_dir)

        # Create figures
        sns.set_style("whitegrid")
        sns.despine()
        self.fig_spectrum_all = plt.figure(figsize=(13, 6))
        self.fig_spectrum_all.suptitle("SPE Spectrum, All Pixels")
        self.fig_spectrum_tm_list = []
        for i in range(32):
            fig = plt.figure(figsize=(13, 6))
            self.fig_spectrum_tm_list.append(plt.figure(figsize=(13, 6)))
        self.fig_combgaus = plt.figure(figsize=(13, 6))
        self.fig_combgaus.suptitle("Combined 1pe fit, All Pixels")
        self.fig_kde = plt.figure(figsize=(13, 6))
        self.fig_kde.suptitle("Distribution of SPE, Kernel density estimate")
        self.fig_hist = plt.figure(figsize=(13, 6))
        self.fig_hist.suptitle("Distribution of SPE, Histogram")

    def start(self):

        # Normalise histogram
        norm = np.sum(np.diff(self.edges, axis=1) * self.hist, axis=1)
        hist = self.hist/norm[:, None]

        # Roll axis for easier plotting
        hist_r = np.rollaxis(hist, 1)
        nbins, npix = hist_r.shape
        e = self.edges[0]
        hist_tops = np.insert(hist_r, np.arange(nbins), hist_r, axis=0)
        edges_tops = np.insert(e, np.arange(e.shape[0]), e, axis=0)[1:-1]

        # Mask dead pixels
        spe = self.dead.mask1d(self.spe)
        spe_sigma = self.dead.mask1d(self.spe_sigma)
        hist_tops = self.dead.mask2d(hist_tops)

        # Spectrum with all pixels
        self.log.info("Plotting: spectrum_all")
        ax_spectrum_all = self.fig_spectrum_all.add_subplot(1, 1, 1)
        ax_spectrum_all.semilogy(edges_tops, hist_tops, color='b', alpha=0.2)
        ax_spectrum_all.set_xlabel("Amplitude (p.e.)")
        ax_spectrum_all.set_ylabel("Probability")

        # Sprectrum for each tm
        self.log.info("Plotting: spectrum_tm")
        hist_tops_tm = np.reshape(hist_tops, (hist_tops.shape[0], 32, 64))
        for tm, fig in enumerate(self.fig_spectrum_tm_list):
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("SPE Spectrum, TM {}".format(tm))
            ax.semilogy(edges_tops, hist_tops_tm[:, tm], color='b', alpha=0.2)
            ax.set_xlabel("Amplitude (p.e.)")
            ax.set_ylabel("Probability")

        # Combined gaussian of each spe value
        self.log.info("Plotting: combined_gaussian")
        ax_comgaus = self.fig_combgaus.add_subplot(1, 1, 1)
        x = np.linspace(-1, 4, 200)
        kernels = []
        for val, sigma in zip(spe.compressed(), spe_sigma.compressed()):
            kernel = stats.norm(val, sigma).pdf(x)
            kernels.append(kernel)
            # plt.plot(x, kernel, color="r")
        sns.rugplot(spe.compressed(), color=".2", linewidth=1, ax=ax_comgaus)
        density = np.sum(kernels, axis=0)
        density /= integrate.trapz(density, x)
        ax_comgaus.plot(x, density)
        ax_comgaus.set_xlabel("SPE Fit Value (p.e.)")
        ax_comgaus.set_ylabel("Sum")

        # Kernel density estimate
        self.log.info("Plotting: spe_kde")
        ax_kde = self.fig_kde.add_subplot(1, 1, 1)
        sns.rugplot(spe.compressed(), color=".2", linewidth=1, ax=ax_kde)
        sns.kdeplot(spe.compressed(), shade=True, ax=ax_kde)
        ax_kde.set_xlabel("SPE Fit Value (p.e.)")
        ax_kde.set_ylabel("KDE")

        # Histogram
        self.log.info("Plotting: histogram")
        ax_hist = self.fig_hist.add_subplot(1, 1, 1)
        sns.distplot(spe.compressed(), kde=False, rug=True, ax=ax_hist)
        ax_hist.set_xlabel("SPE Fit Value (p.e.)")
        ax_hist.set_ylabel("N")

    def finish(self):
        output_path = join(self.output_dir, "spectrum_all.png")
        self.fig_spectrum_all.savefig(output_path)
        self.log.info("Created figure: {}".format(output_path))

        output_path = join(self.output_dir, "spectrum_tm{}.png")
        for tm, fig in enumerate(self.fig_spectrum_tm_list):
            p = output_path.format(tm)
            fig.savefig(p)
            self.log.info("Created figure: {}".format(p))

        output_path = join(self.output_dir, "combined_gaussian.png")
        self.fig_combgaus.savefig(output_path)
        self.log.info("Created figure: {}".format(output_path))

        output_path = join(self.output_dir, "kde.png")
        self.fig_kde.savefig(output_path)
        self.log.info("Created figure: {}".format(output_path))

        output_path = join(self.output_dir, "hist.png")
        self.fig_hist.savefig(output_path)
        self.log.info("Created figure: {}".format(output_path))

exe = ADC2PEResidualsPlotter()
exe.run()
