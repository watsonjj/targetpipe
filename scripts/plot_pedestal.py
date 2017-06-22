"""
Create a pedestal file from an event file using the target_calib Pedestal
class
"""

from traitlets import Dict, List
from ctapipe.core import Tool, Component
from ctapipe.io.eventfilereader import EventFileReaderFactory
from targetpipe.calib.camera.makers import PedestalMaker
from targetpipe.plots.official import OfficialPlotter
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
# import seaborn as sns
from os.path import join


class HitsPlotter(OfficialPlotter):
    name = 'HitsPlotter'

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
        super().__init__(config=config, tool=tool, **kwargs)

        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(1, 1, 1)

    def create(self, value, value_title):
        n_blks, n_bps = value.shape
        pixel_hits_0 = np.ma.masked_where(value == 0, value)

        im = self.ax.pcolor(pixel_hits_0, cmap="viridis", edgecolors='white', linewidths=0.1)
        cbar = self.fig.colorbar(im)
        self.ax.patch.set(hatch='xx')
        self.ax.set_xlabel("Blockphase + Waveform position")
        self.ax.set_ylabel("Block")
        cbar.set_label(value_title)


class PedestalBuilder(Tool):
    name = "PedestalBuilder"
    description = "Create the TargetCalib Pedestal file from waveforms"

    aliases = Dict(dict(f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ))
    flags = Dict(dict())
    classes = List([EventFileReaderFactory,
                    PedestalMaker
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reader = None
        self.pedmaker = None

        self.bps = None

        self.p_hits = None
        self.p_pedestalpix2d = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.reader = reader_class(**kwargs)

        first_event = self.reader.get_event(0)
        n_modules = first_event.meta['n_modules']
        n_blocks = first_event.meta['n_blocks']
        n_samples = first_event.r0.tel[0].adc_samples.shape[2]

        ped_path = self.reader.input_path.replace("_r0.tio", "_ped.tcal")

        self.pedmaker = PedestalMaker(**kwargs,
                                      output_path=ped_path,
                                      n_tms=n_modules,
                                      n_blocks=n_blocks,
                                      n_samples=n_samples)

        script = "plot_pedestal_hits"
        self.p_hits = HitsPlotter(**kwargs, shape="wide")
        self.p_pedestalpix2d = HitsPlotter(**kwargs, shape="wide")

    def start(self):
        n_events = self.reader.num_events

        self.bps = np.zeros(n_events)

        desc = "Filling pedestal"
        source = self.reader.read()
        for event in tqdm(source, total=n_events, desc=desc):
            ev = event.count
            self.bps[ev] = event.r0.tel[0].blockphase[0]
            self.pedmaker.add_event(event)

    def finish(self):
        self.log.info("Extracting hits into numpy array")
        hits = np.array(self.pedmaker.ped_obj.GetHits())
        self.log.info("Extracting pedestal into numpy array")
        pedestal = np.array(self.pedmaker.ped_obj.GetPed())

        minimum = hits.min()
        maximum = hits.max()
        minimum_non_zero = hits[hits.nonzero()].min()
        self.log.info("Minimum hits = {}".format(minimum))
        self.log.info("Minimum non-zero hits = {}".format(minimum_non_zero))
        self.log.info("Maximum hits = {}".format(maximum))

        bps = np.unique(self.bps)
        self.log.info("Blockphases used: {}".format(bps))

        # Find used pixel
        tmpix_sum = np.sum(hits, axis=(2, 3))
        tm_sum = np.sum(tmpix_sum, axis=1)
        tm = tm_sum.argmax()
        tmpix = tmpix_sum[tm].argmax()

        self.log.info("Using tm {} tmpix {}".format(tm, tmpix))
        pixel_hits = hits[tm, tmpix]
        pixel_pedestal = pedestal[tm, tmpix]

        self.p_hits.create(pixel_hits, "Hits")
        self.p_pedestalpix2d.create(pixel_pedestal, "Pedestal (ADC)")

        output_dir = join(self.reader.output_directory, "plot_pedestal")

        output_path = join(output_dir, "hits.pdf")
        self.p_hits.save(output_path)

        output_path = join(output_dir, "pedestal.pdf")
        self.p_pedestalpix2d.save(output_path)

exe = PedestalBuilder()
exe.run()
