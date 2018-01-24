import matplotlib as mpl
mpl.use('pgf')

from ctapipe.core import Component
from traitlets import CaselessStrEnum as CaStEn, Unicode
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from os.path import join, exists, dirname, splitext
from os import makedirs
import numpy as np
import seaborn as sns


class ChecmPaperPlotter(Component):
    type = CaStEn(['paper', 'talk'], 'paper',
                  help="Intended publishment of plot").tag(config=True)
    shape = CaStEn(['square', 'wide'], 'square',
                   help="Shape of plot").tag(config=True)
    script = Unicode("", allow_none=True,
                     help='Name of the calling script').tag(config=True)
    figure_name = Unicode("", allow_none=True,
                          help='Figure name').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, parent=tool, **kwargs)

        sns.set_style("white")
        sns.set_style("ticks")
        sns.set_context(self.type, rc={"font.famly": "Helvetica",
                                       "font.size":10,
                                       "axes.titlesize":10,
                                       "axes.labelsize":10,
                                       "legend.fontsize": 10,
                                       "text.fontsize": 10,
                                       "lines.markeredgewidth": 1,
                                       })

        self.fig, self.ax = self.create_figure()


        self.ax.xaxis.set_minor_locator(AutoMinorLocator())
        self.ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.tick_params(which='both', width=1)
        plt.tick_params(which='minor', length=4)
        plt.tick_params(which='major', length=7)

        # self.ax.tick_params(labelsize=19)

        self.extension = 'pdf'
        self.base_dir = "/Volumes/gct-jason/plots/checm_paper"

    @property
    def output_dir(self):
        return join(self.base_dir, self.script)

    @property
    def output_path(self):
        return join(self.output_dir, self.figure_name + "." + self.extension)

    def create_figure(self):
        if self.shape == 'wide':
            fig = plt.figure(figsize=(8, 4))
        else:
            fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1)
        return fig, ax

    def save(self, output_path=None):
        if output_path:
            output_dir = dirname(output_path)
        else:
            output_path = self.output_path
            output_dir = self.output_dir

        if not exists(output_dir):
            self.log.info("Creating directory: {}".format(output_dir))
            makedirs(output_dir)

        self.fig.savefig(output_path, bbox_inches='tight')
        self.log.info("Figure saved to: {}".format(output_path))


class Plotter():
    def __init__(self, context, base_dir, script, figure_name):
        self.context = context
        self.base_dir = base_dir
        self.script = script
        self.figure_name = figure_name

        # sns.set_style("white")
        # sns.set_style("ticks")
        rc = {  # setup matplotlib to use latex for output
            "pgf.texsystem": "pdflatex", # change this if using xetex or lautex
            "text.usetex": True,         # use LaTeX to write all text
            "font.family": "cursive",
            "font.serif": [],            # blank entries should cause plots to inherit fonts from the document
            "font.sans-serif": [],
            "font.monospace": [],
            "font.cursive": [],
            "axes.titlesize": 10,
            "axes.labelsize": 10,        # LaTeX default is 10pt font.
            "font.size": 10,
            "legend.fontsize": 8,        # Make the legend/label fonts a little smaller
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.figsize": self.figsize(0.9), # default fig size of 0.9 textwidth
            "lines.markeredgewidth": 1,
            "pgf.preamble": [
                r"\usepackage[utf8x]{inputenc}", # use utf8 fonts becasue your computer can handle it :)
                r"\usepackage[T1]{fontenc}" # plots will be generated using this preamble
            ]
        }
        mpl.rcParams.update(rc)
        sns.set_context(self.context, rc=rc)

        self.fig, self.ax = self.create_figure()

    @property
    def output_dir(self):
        return join(self.base_dir, self.script)

    @property
    def output_path_pdf(self):
        return join(self.output_dir, self.figure_name + "." + "pdf")

    @property
    def output_path_pgf(self):
        return join(self.output_dir, self.figure_name + "." + "pgf")

    @staticmethod
    def figsize(scale=0.9):
        fig_width_pt = 469.755  # Get this from LaTeX using \the\textwidth
        inches_per_pt = 1.0 / 72.27  # Convert pt to inch
        golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
        fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
        fig_height = fig_width * golden_mean  # height in inches
        fig_size = [fig_width, fig_height]
        return fig_size

    def create_figure(self):
        fig = plt.figure(figsize=self.figsize())
        ax = fig.add_subplot(1, 1, 1)

        fmt = mpl.ticker.StrMethodFormatter("{x}")
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        return fig, ax

    def add_legend(self, loc="upper right"):
        self.ax.legend(loc=loc)
        self.ax.legend.set_linewidth(15)

    @staticmethod
    def create_directory(directory):
        if not exists(directory):
            print("Creating directory: {}".format(directory))
            makedirs(directory)

    def save(self, output_path=None):
        if output_path:
            output_dir = dirname(output_path)
            self.create_directory(output_dir)
            path, ext = splitext(output_path)
            if ext and (ext != ".pdf" or ext != "pgf"):
                self.fig.savefig(output_path, bbox_inches='tight')
                print("Figure saved to: {}".format(output_path))
            else:
                path_pdf = path + ".pdf"
                path_pgf = path + ".pgf"
                self.fig.savefig(path_pdf, bbox_inches='tight')
                print("Figure saved to: {}".format(path_pdf))
                self.fig.savefig(path_pgf, bbox_inches='tight')
                print("Figure saved to: {}".format(path_pgf))
        else:
            self.create_directory(self.output_dir)
            self.fig.savefig(self.output_path_pdf, bbox_inches='tight')
            print("Figure saved to: {}".format(self.output_path_pdf))
            self.fig.savefig(self.output_path_pgf, bbox_inches='tight')
            print("Figure saved to: {}".format(self.output_path_pgf))


class ThesisPlotter(Plotter):
    def __init__(self, script, figure_name):
        type = 'paper'
        base_dir = "/Users/Jason/Dropbox/DropboxDocuments/University/Oxford/Reports/Thesis/figures/plots"
        super().__init__(type, base_dir, script, figure_name)
