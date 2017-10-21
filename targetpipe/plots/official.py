from matplotlib.ticker import AutoMinorLocator

from ctapipe.core import Component
from traitlets import CaselessStrEnum as CaStEn, Unicode
from matplotlib import pyplot as plt
import seaborn as sns
from os.path import join, exists, dirname
from os import makedirs


class OfficialPlotter(Component):
    name = 'Official'

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
                                       "text.fontsize": 10
                                       })

        self.fig, self.ax = self.create_figure()

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


class ChecmPaperPlotter(OfficialPlotter):
    name = 'ChecmPaperPlotter'

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

        self.ax.xaxis.set_minor_locator(AutoMinorLocator())
        self.ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.tick_params(which='both', width=1)
        plt.tick_params(which='minor', length=4)
        plt.tick_params(which='major', length=7)

        # self.ax.tick_params(labelsize=19)

        self.base_dir = "/Volumes/gct-jason/plots/checm_paper"


class ThesisPlotter(ChecmPaperPlotter):
    name = 'ThesisPlotter'

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

        self.base_dir = "/Users/Jason/Dropbox/DropboxDocuments/University/Oxford/Reports/Thesis/plots"
