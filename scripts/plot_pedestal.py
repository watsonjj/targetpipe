"""
Create a pedestal file from an event file using the target_calib Pedestal
class
"""

from traitlets import Dict, List, Bool
from ctapipe.core import Tool, Component
from ctapipe.io.eventfilereader import EventFileReaderFactory
from targetpipe.calib.camera.makers import PedestalMaker
from targetpipe.calib.camera.r1 import TargetioR1Calibrator
from targetpipe.plots.official import OfficialPlotter
from target_calib import Calibrator
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
# import seaborn as sns
from os.path import join
from IPython import embed
from matplotlib.colors import LogNorm


class BlockPlotter(OfficialPlotter):
    name = 'BlockPlotter'

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

        # self.ax.set_ylim(110, 120)


class ResidualPlotter(OfficialPlotter):
    name = 'ResidualPlotter'

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

        self.hist = None
        self.xedges = None
        self.yedges = None

    def initialize(self, x_n, x_range, y_n=200, y_range=(-16, 16)):
        np_hist = np.histogram2d([0], [0], bins=[x_n, y_n], range=[x_range, y_range])
        self.hist, self.xedges, self.yedges = np_hist
        self.hist[:] = 0

    def add(self, x, y):
        hist, _, _ = np.histogram2d(x, y, bins=[self.xedges, self.yedges])
        self.hist += hist

    def create(self, x_label, y_label, mean, stddev):
        self.hist = np.ma.masked_where(self.hist == 0, self.hist)
        # area = np.diff(self.xedges)[:, None] * np.diff(self.yedges)[None, :]
        # volume = self.hist * area
        z = self.hist# / volume.sum()
        im = self.ax.pcolormesh(self.xedges, self.yedges, z.T, cmap="viridis", edgecolors='white', linewidths=0, norm=LogNorm(vmin=self.hist.min(), vmax=self.hist.max()))
        cbar = self.fig.colorbar(im)
        self.ax.axhline(mean, linewidth=0.1, color='r')
        self.ax.axhline(mean+stddev, linewidth=0.1, color='orange')
        self.ax.axhline(mean-stddev, linewidth=0.1, color='orange')
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        cbar.set_label("N")


class ResidualStatsPlotter(OfficialPlotter):
    name = 'ResidualStatsPlotter'

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

    def create(self, x, x_n, y, y_n, x_label, y_label):
        x_range = (x.min(), x.max())
        y_range = (y.min(), y.max())
        hist, xedges, yedges = np.histogram2d(x, y, bins=[x_n, y_n], range=[x_range, y_range])
        hist = np.ma.masked_where(hist == 0, hist)
        z = hist
        im = self.ax.pcolormesh(xedges, yedges, z.T, cmap="viridis",
                                edgecolors='white', linewidths=0,
                                norm=LogNorm(vmin=hist.min(), vmax=hist.max()))
        cbar = self.fig.colorbar(im)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        cbar.set_label("N")


class PedestalBuilder(Tool):
    name = "PedestalBuilder"
    description = "Create the TargetCalib Pedestal file from waveforms"

    residual_flag = Bool(False, help="Create residual plots?").tag(config=True)

    aliases = Dict(dict(f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ))
    flags = Dict(dict(R=({'PedestalBuilder': {'residual_flag': True}},
                         "Create residual plots?")
                      ))
    classes = List([EventFileReaderFactory,
                    PedestalMaker
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reader = None
        self.pedmaker = None

        self.bps = None

        self.n_modules = None
        self.n_blocks = None
        self.n_cells = None
        self.n_rows = None
        self.n_columns = None
        self.n_blockphases = None
        self.n_pixels = None
        self.n_samples = None

        self.output_dir = None
        self.ped_path = None

        self.residual_mean = None
        self.residual_stddev = None

        self.p_hits = None
        self.p_pedestalpix2d = None
        self.p_res_eventindex = None
        self.p_res_pixel = None
        self.p_res_tm = None
        self.p_res_tmpix = None
        self.p_res_fci = None
        self.p_res_blk = None
        self.p_res_row = None
        self.p_res_col = None
        self.p_res_bph = None
        self.p_avg_res_eventindex = None
        self.p_avg_res_fci = None
        self.p_avg_res_blk = None
        self.p_avg_res_row = None
        self.p_avg_res_col = None
        self.p_avg_res_bph = None
        self.p_std_res_eventindex = None
        self.p_std_res_fci = None
        self.p_std_res_blk = None
        self.p_std_res_row = None
        self.p_std_res_col = None
        self.p_std_res_bph = None
        self.p_min_res_eventindex = None
        self.p_min_res_fci = None
        self.p_min_res_blk = None
        self.p_min_res_row = None
        self.p_min_res_col = None
        self.p_min_res_bph = None
        self.p_max_res_eventindex = None
        self.p_max_res_fci = None
        self.p_max_res_blk = None
        self.p_max_res_row = None
        self.p_max_res_col = None
        self.p_max_res_bph = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.reader = reader_class(**kwargs)

        first_event = self.reader.get_event(0)
        self.n_modules = first_event.meta['n_modules']
        self.n_blocks = first_event.meta['n_blocks']
        self.n_cells = first_event.meta['n_cells']
        self.n_rows = first_event.meta['n_rows']
        self.n_columns = first_event.meta['n_columns']
        self.n_blockphases = first_event.meta['n_blockphases']
        self.n_pixels, self.n_samples = first_event.r0.tel[0].adc_samples[0].shape

        self.output_dir = join(self.reader.output_directory, "plot_pedestal")

        ped_name = self.reader.filename.replace("_r0", "_ped.tcal")
        self.ped_path = join(self.output_dir, ped_name)

        self.pedmaker = PedestalMaker(**kwargs,
                                      output_path=self.ped_path,
                                      n_tms=self.n_modules,
                                      n_blocks=self.n_blocks,
                                      n_samples=self.n_samples)

        script = "plot_pedestal_hits"
        self.p_hits = BlockPlotter(**kwargs, shape="wide")
        self.p_pedestalpix2d = BlockPlotter(**kwargs, shape="wide")
        self.p_res_eventindex = ResidualPlotter(**kwargs, shape="wide")
        self.p_res_pixel = ResidualPlotter(**kwargs, shape="wide")
        self.p_res_tm = ResidualPlotter(**kwargs, shape="wide")
        self.p_res_tmpix = ResidualPlotter(**kwargs, shape="wide")
        self.p_res_fci = ResidualPlotter(**kwargs, shape="wide")
        self.p_res_blk = ResidualPlotter(**kwargs, shape="wide")
        self.p_res_row = ResidualPlotter(**kwargs, shape="wide")
        self.p_res_col = ResidualPlotter(**kwargs, shape="wide")
        self.p_res_bph = ResidualPlotter(**kwargs, shape="wide")
        self.p_avg_res_eventindex = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_avg_res_fci = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_avg_res_blk = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_avg_res_row = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_avg_res_col = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_avg_res_bph = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_std_res_eventindex = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_std_res_fci = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_std_res_blk = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_std_res_row = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_std_res_col = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_std_res_bph = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_min_res_eventindex = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_min_res_fci = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_min_res_blk = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_min_res_row = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_min_res_col = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_min_res_bph = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_max_res_eventindex = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_max_res_fci = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_max_res_blk = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_max_res_row = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_max_res_col = ResidualStatsPlotter(**kwargs, shape="wide")
        self.p_max_res_bph = ResidualStatsPlotter(**kwargs, shape="wide")

    def start(self):
        n_events = self.reader.num_events
        self.bps = np.zeros(n_events)

        desc = "Filling pedestal"
        source = self.reader.read()
        for event in tqdm(source, total=n_events, desc=desc):
            ev = event.count
            self.bps[ev] = event.r0.tel[0].blockphase[0]
            self.pedmaker.add_event(event)

        if self.residual_flag:
            self.pedmaker.save()
            kwargs = dict(config=self.config, tool=self,
                          pedestal_path=self.ped_path)
            r1_calibrator = TargetioR1Calibrator(**kwargs)

            size = 500
            shape = (size, self.n_pixels, self.n_samples)
            r1_container = np.zeros(shape)
            event_arr = np.zeros(shape)
            pixel_arr = np.arange(self.n_pixels)[None, :, None] * np.ones(shape)
            tm_arr = (np.arange(self.n_pixels)[None, :, None] // 64) * np.ones(shape)
            tmpix_arr = (np.arange(self.n_pixels)[None, :, None] % 64) * np.ones(shape)
            fci_arr = np.zeros(shape)
            blk_arr = np.zeros(shape)
            row_arr = np.zeros(shape)
            col_arr = np.zeros(shape)
            bph_arr = np.zeros(shape)

            event_mean = np.zeros(n_events)
            event_std = np.zeros(n_events)
            event_min = np.zeros(n_events)
            event_max = np.zeros(n_events)
            event_fci = np.zeros(n_events)
            event_blk = np.zeros(n_events)
            event_row = np.zeros(n_events)
            event_col = np.zeros(n_events)
            event_bph = np.zeros(n_events)

            self.p_res_eventindex.initialize(500, (0, n_events))
            self.p_res_pixel.initialize(self.n_pixels, (0, self.n_pixels))
            self.p_res_tm.initialize(self.n_modules, (0, self.n_modules))
            self.p_res_tmpix.initialize(64, (0, 64))
            self.p_res_fci.initialize(500, (0, self.n_cells))
            self.p_res_blk.initialize(self.n_blocks, (0, self.n_blocks))
            self.p_res_row.initialize(self.n_rows, (0, self.n_rows))
            self.p_res_col.initialize(self.n_columns, (0, self.n_columns))
            self.p_res_bph.initialize(self.n_blockphases, (0, self.n_blockphases))

            desc = "Extracting residuals"
            source = self.reader.read()
            count = 0
            for event in tqdm(source, total=n_events, desc=desc):
                ev = event.count

                if (ev > 0) & (ev % size == 0):
                    self.p_res_eventindex.add(event_arr.ravel(), r1_container.ravel())
                    self.p_res_pixel.add(pixel_arr.ravel(), r1_container.ravel())
                    self.p_res_tm.add(tm_arr.ravel(), r1_container.ravel())
                    self.p_res_tmpix.add(tmpix_arr.ravel(), r1_container.ravel())
                    self.p_res_fci.add(fci_arr.ravel(), r1_container.ravel())
                    self.p_res_blk.add(blk_arr.ravel(), r1_container.ravel())
                    self.p_res_row.add(row_arr.ravel(), r1_container.ravel())
                    self.p_res_col.add(col_arr.ravel(), r1_container.ravel())
                    self.p_res_bph.add(bph_arr.ravel(), r1_container.ravel())
                    count = 0

                r1_calibrator.calibrate(event)
                r1 = event.r1.tel[0].pe_samples[0]

                r1_container[ev%size] = r1
                event_arr[ev%size, :, :] = ev
                fci = event.r0.tel[0].first_cell_ids[0]
                r = event.r0.tel[0].row[0]
                c = event.r0.tel[0].column[0]
                blk = c * 8 + r
                bph = event.r0.tel[0].blockphase[0]
                fci_arr[ev%size, :, :] = fci
                blk_arr[ev%size, :, :] = blk
                row_arr[ev%size, :, :] = r
                col_arr[ev%size, :, :] = c
                bph_arr[ev%size, :, :] = bph
                event_fci[ev] = fci
                event_blk[ev] = blk
                event_row[ev] = r
                event_col[ev] = c
                event_bph[ev] = bph

                event_mean[ev] = np.mean(r1)
                event_std[ev] = np.std(r1)
                event_min[ev] = np.min(r1)
                event_max[ev] = np.max(r1)

                count += 1

            self.p_res_eventindex.add(event_arr[:count].ravel(), r1_container[:count].ravel())
            self.p_res_pixel.add(pixel_arr[:count].ravel(), r1_container[:count].ravel())
            self.p_res_tm.add(tm_arr[:count].ravel(), r1_container[:count].ravel())
            self.p_res_tmpix.add(tmpix_arr[:count].ravel(), r1_container[:count].ravel())
            self.p_res_fci.add(fci_arr[:count].ravel(), r1_container[:count].ravel())
            self.p_res_blk.add(blk_arr[:count].ravel(), r1_container[:count].ravel())
            self.p_res_row.add(row_arr[:count].ravel(), r1_container[:count].ravel())
            self.p_res_col.add(col_arr[:count].ravel(), r1_container[:count].ravel())
            self.p_res_bph.add(bph_arr[:count].ravel(), r1_container[:count].ravel())

            self.p_avg_res_eventindex.create(np.arange(n_events), 500, event_mean, 200, "Event Index", "Residual ADC Samples (Event Average)")
            self.p_avg_res_fci.create(event_fci, 500, event_mean, 200, "First Cell ID", "Residual ADC Samples (Event Average)")
            self.p_avg_res_blk.create(event_blk, self.n_blocks, event_mean, 200, "Block", "Residual ADC Samples (Event Average)")
            self.p_avg_res_row.create(event_row, self.n_rows, event_mean, 200, "Row", "Residual ADC Samples (Event Average)")
            self.p_avg_res_col.create(event_col, self.n_columns, event_mean, 200, "Column", "Residual ADC Samples (Event Average)")
            self.p_avg_res_bph.create(event_bph, self.n_blockphases, event_mean, 200, "Blockphase", "Residual ADC Samples (Event Average)")
            self.p_std_res_eventindex.create(np.arange(n_events), 500, event_std, 200, "Event Index", "Residual ADC Samples (Event Standard Deviation)")
            self.p_std_res_fci.create(event_fci, 500, event_std, 200, "First Cell ID", "Residual ADC Samples (Event Standard Deviation)")
            self.p_std_res_blk.create(event_blk, self.n_blocks, event_std, 200, "Block", "Residual ADC Samples (Event Standard Deviation)")
            self.p_std_res_row.create(event_row, self.n_rows, event_std, 200, "Row", "Residual ADC Samples (Event Standard Deviation)")
            self.p_std_res_col.create(event_col, self.n_columns, event_std, 200, "Column", "Residual ADC Samples (Event Standard Deviation)")
            self.p_std_res_bph.create(event_bph, self.n_blockphases, event_std, 200, "Blockphase", "Residual ADC Samples (Event Standard Deviation)")
            self.p_min_res_eventindex.create(np.arange(n_events), 500, event_min, 200, "Event Index", "Residual ADC Samples (Event Minimum)")
            self.p_min_res_fci.create(event_fci, 500, event_min, 200, "First Cell ID", "Residual ADC Samples (Event Minimum)")
            self.p_min_res_blk.create(event_blk, self.n_blocks, event_min, 200, "Block", "Residual ADC Samples (Event Minimum)")
            self.p_min_res_row.create(event_row, self.n_rows, event_min, 200, "Row", "Residual ADC Samples (Event Minimum)")
            self.p_min_res_col.create(event_col, self.n_columns, event_min, 200, "Column", "Residual ADC Samples (Event Minimum)")
            self.p_min_res_bph.create(event_bph, self.n_blockphases, event_min, 200, "Blockphase", "Residual ADC Samples (Event Minimum)")
            self.p_max_res_eventindex.create(np.arange(n_events), 500, event_max, 200, "Event Index", "Residual ADC Samples (Event Maximum)")
            self.p_max_res_fci.create(event_fci, 500, event_max, 200, "First Cell ID", "Residual ADC Samples (Event Maximum)")
            self.p_max_res_blk.create(event_blk, self.n_blocks, event_max, 200, "Block", "Residual ADC Samples (Event Maximum)")
            self.p_max_res_row.create(event_row, self.n_rows, event_max, 200, "Row", "Residual ADC Samples (Event Maximum)")
            self.p_max_res_col.create(event_col, self.n_columns, event_max, 200, "Column", "Residual ADC Samples (Event Maximum)")
            self.p_max_res_bph.create(event_bph, self.n_blockphases, event_max, 200, "Blockphase", "Residual ADC Samples (Event Maximum)")

            ps = self.n_pixels * self.n_samples
            self.residual_mean = np.sum(ps * event_mean) / (n_events * ps)
            self.residual_stddev = ((np.sum(ps * event_std ** 2) + np.sum(ps * (event_mean - self.residual_mean) ** 2)) / (n_events * ps)) ** 0.5

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
        output_path = join(self.output_dir, "hits.pdf")
        self.p_hits.save(output_path)

        self.p_pedestalpix2d.create(pixel_pedestal, "Pedestal (ADC)")
        output_path = join(self.output_dir, "pedestal.pdf")
        self.p_pedestalpix2d.save(output_path)

        if self.residual_flag:
            self.p_res_eventindex.create("Event Index", "Residual ADC Samples", self.residual_mean, self.residual_stddev)
            output_path = join(self.output_dir, "residual_eventindex.pdf")
            self.p_res_eventindex.save(output_path)

            self.p_res_pixel.create("Pixel", "Residual ADC Samples", self.residual_mean, self.residual_stddev)
            output_path = join(self.output_dir, "residual_pixel.pdf")
            self.p_res_pixel.save(output_path)

            self.p_res_tm.create("TM", "Residual ADC Samples", self.residual_mean, self.residual_stddev)
            output_path = join(self.output_dir, "residual_tm.pdf")
            self.p_res_tm.save(output_path)

            self.p_res_tmpix.create("TMPIX", "Residual ADC Samples", self.residual_mean, self.residual_stddev)
            output_path = join(self.output_dir, "residual_tmpix.pdf")
            self.p_res_tmpix.save(output_path)

            self.p_res_fci.create("First Cell ID", "Residual ADC Samples", self.residual_mean, self.residual_stddev)
            output_path = join(self.output_dir, "residual_fci.pdf")
            self.p_res_fci.save(output_path)

            self.p_res_blk.create("Block", "Residual ADC Samples", self.residual_mean, self.residual_stddev)
            output_path = join(self.output_dir, "residual_blk.pdf")
            self.p_res_blk.save(output_path)

            self.p_res_row.create("Row", "Residual ADC Samples", self.residual_mean, self.residual_stddev)
            output_path = join(self.output_dir, "residual_row.pdf")
            self.p_res_row.save(output_path)

            self.p_res_col.create("Column", "Residual ADC Samples", self.residual_mean, self.residual_stddev)
            output_path = join(self.output_dir, "residual_col.pdf")
            self.p_res_col.save(output_path)

            self.p_res_bph.create("Blockphase", "Residual ADC Samples", self.residual_mean, self.residual_stddev)
            output_path = join(self.output_dir, "residual_bph.pdf")
            self.p_res_bph.save(output_path)

            output_path = join(self.output_dir, "avg_residual_eventindex.pdf")
            self.p_avg_res_eventindex.save(output_path)

            output_path = join(self.output_dir, "avg_residual_fci.pdf")
            self.p_avg_res_fci.save(output_path)

            output_path = join(self.output_dir, "avg_residual_blk.pdf")
            self.p_avg_res_blk.save(output_path)

            output_path = join(self.output_dir, "avg_residual_row.pdf")
            self.p_avg_res_row.save(output_path)

            output_path = join(self.output_dir, "avg_residual_col.pdf")
            self.p_avg_res_col.save(output_path)

            output_path = join(self.output_dir, "avg_residual_bph.pdf")
            self.p_avg_res_bph.save(output_path)

            output_path = join(self.output_dir, "std_residual_eventindex.pdf")
            self.p_std_res_eventindex.save(output_path)

            output_path = join(self.output_dir, "std_residual_fci.pdf")
            self.p_std_res_fci.save(output_path)

            output_path = join(self.output_dir, "std_residual_blk.pdf")
            self.p_std_res_blk.save(output_path)

            output_path = join(self.output_dir, "std_residual_row.pdf")
            self.p_std_res_row.save(output_path)

            output_path = join(self.output_dir, "std_residual_col.pdf")
            self.p_std_res_col.save(output_path)

            output_path = join(self.output_dir, "std_residual_bph.pdf")
            self.p_std_res_bph.save(output_path)

            output_path = join(self.output_dir, "min_residual_eventindex.pdf")
            self.p_min_res_eventindex.save(output_path)

            output_path = join(self.output_dir, "min_residual_fci.pdf")
            self.p_min_res_fci.save(output_path)

            output_path = join(self.output_dir, "min_residual_blk.pdf")
            self.p_min_res_blk.save(output_path)

            output_path = join(self.output_dir, "min_residual_row.pdf")
            self.p_min_res_row.save(output_path)

            output_path = join(self.output_dir, "min_residual_col.pdf")
            self.p_min_res_col.save(output_path)

            output_path = join(self.output_dir, "min_residual_bph.pdf")
            self.p_min_res_bph.save(output_path)

            output_path = join(self.output_dir, "max_residual_eventindex.pdf")
            self.p_max_res_eventindex.save(output_path)

            output_path = join(self.output_dir, "max_residual_fci.pdf")
            self.p_max_res_fci.save(output_path)

            output_path = join(self.output_dir, "max_residual_blk.pdf")
            self.p_max_res_blk.save(output_path)

            output_path = join(self.output_dir, "max_residual_row.pdf")
            self.p_max_res_row.save(output_path)

            output_path = join(self.output_dir, "max_residual_col.pdf")
            self.p_max_res_col.save(output_path)

            output_path = join(self.output_dir, "max_residual_bph.pdf")
            self.p_max_res_bph.save(output_path)

exe = PedestalBuilder()
exe.run()
