from matplotlib.ticker import AutoMinorLocator

from ctapipe.core import Component
from traitlets import CaselessStrEnum as CaStEn, Unicode
from matplotlib import pyplot as plt
import seaborn as sns
from os.path import join, exists, dirname
from os import makedirs


class OfficialPlotter(Component):
    name = 'OfficialPlotter'

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
                                       "font.size":14,
                                       "axes.titlesize":14,
                                       "axes.labelsize":14
                                       })

        if self.shape == 'wide':
            self.fig = plt.figure(figsize=(7, 5))
        else:
            self.fig = plt.figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        # plt.tick_params(which='both', width=1)
        # plt.tick_params(which='minor', length=4)
        # plt.tick_params(axis='both', which='major', length=8)

        self.extension = 'pdf'
        base_dir = "/Volumes/gct-jason/plots/checm_paper"
        self.output_dir = join(base_dir, self.script)
        figure_name = self.figure_name + "." + self.extension
        self.output_path = join(self.output_dir, figure_name)

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
