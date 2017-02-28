import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from traitlets import Dict, List, Unicode
from ctapipe.core import Tool, Component
from ctapipe.io.eventfilereader import EventFileReaderFactory
from targetpipe.calib.camera.pedestal import PedestalSubtractor


class StepPlotter(Component):
    name = 'StepPlotter'

    output_dir = Unicode('./outputs', allow_none=True,
                         help='Output path to the directory where the plots '
                              'will be saved. Default: a directory is created '
                              'in the current directory').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        Plotter for camera images.

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
        self._init_figure()

    def _init_figure(self):
        plt.style.use("ggplot")
        self.fig = plt.figure(figsize=(13, 6))
        self.wav_raw_ax = self.fig.add_subplot(3, 2, 1)
        self.hist_raw_ax = self.fig.add_subplot(3, 2, 2)
        self.wav_subf_ax = self.fig.add_subplot(3, 2, 3)
        self.hist_subf_ax = self.fig.add_subplot(3, 2, 4)
        self.wav_subi_ax = self.fig.add_subplot(3, 2, 5)
        self.hist_subi_ax = self.fig.add_subplot(3, 2, 6)

        plt.tight_layout()
        plt.subplots_adjust(left=0.07, top=0.95, bottom=0.1,
                            hspace=0.6, wspace=0.2)

        self.wav_raw_ax.set_title("Raw")
        self.wav_subf_ax.set_title("Float Subtraction")
        self.wav_subi_ax.set_title("UInt16 Subtraction")

        self.wav_raw_ax.set_xlabel("Time (ns)")
        self.wav_raw_ax.set_ylabel("ADC")
        self.wav_subf_ax.set_xlabel("Time (ns)")
        self.wav_subf_ax.set_ylabel("Sample (ADC)")
        self.wav_subi_ax.set_xlabel("Time (ns)")
        self.wav_subi_ax.set_ylabel("ADC")
        self.hist_raw_ax.set_xlabel("Sample (ADC)")
        self.hist_raw_ax.set_ylabel("N_samples")
        self.hist_subf_ax.set_xlabel("Sample (ADC)")
        self.hist_subf_ax.set_ylabel("N_samples")
        self.hist_subi_ax.set_xlabel("Sample (ADC)")
        self.hist_subi_ax.set_ylabel("N_samples")

    def plot(self, waveforms, pedsub_f, pedsub_i, image_path):
        n_plots = 10

        n_events = waveforms.shape[0]
        n_pix = waveforms.shape[1]

        haspixdata = (waveforms.sum(axis=0).sum(axis=1) != 0)
        p = np.arange(n_pix)[haspixdata]
        e = np.arange(n_events)

        # Choose which waveforms to plot
        # Different Pixel, Same waveforms
        events = list(np.random.choice(e, size=1))
        pixels = list(np.random.choice(p, size=n_plots))

        for ev in events:
            for pix in pixels:
                self.wav_raw_ax.plot(waveforms[ev, pix])
                self.wav_subf_ax.plot(pedsub_f[ev, pix])
                self.wav_subi_ax.plot(pedsub_i[ev, pix])

        hasdata_wf = waveforms[:, haspixdata]
        hasdata_pedsub_f = pedsub_f[:, haspixdata]
        hasdata_pedsub_i = pedsub_i[:, haspixdata]

        self.hist_raw_ax.hist(hasdata_wf.flatten(), 100, alpha=0.75)
        self.hist_subf_ax.hist(hasdata_pedsub_f.flatten(), 100, alpha=0.75)
        self.hist_subi_ax.hist(hasdata_pedsub_i.flatten(), 100, alpha=0.75)

        self.hist_raw_ax.set_yscale('log')
        self.hist_subf_ax.set_yscale('log')
        self.hist_subi_ax.set_yscale('log')

        self.log.info("Saving waveforms plot to: {}".format(image_path))
        self.fig.savefig(image_path, format='pdf', bbox_inches='tight')


class TargetCalibStepDisplayer(Tool):
    name = "TargetCalibStepDisplayer"
    description = "Plot the calibration steps used in TargetCalib"

    output_dir = Unicode(None, allow_none=True,
                         help='Output path to the directory where the plots '
                              'will be saved. If None, a directory is created '
                              'in the location of the '
                              'input file.').tag(config=True)

    aliases = Dict(dict(f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        P='PedestalSubtractor.pedestal_path',
                        O='TargetCalibStepDisplayer.output_dir',
                        ))
    classes = List([EventFileReaderFactory,
                    StepPlotter,
                    PedestalSubtractor
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_reader = None
        self.extractor = None
        self.calibrator = None
        self.plotter = None
        self.ped = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.file_reader = reader_class(**kwargs)

        self.plotter = StepPlotter(**kwargs)

        self.ped = PedestalSubtractor(**kwargs)

    def start(self):
        # Open file
        telid = 0
        n_events = self.file_reader.num_events
        first_event = self.file_reader.get_event(0)

        # Setup sizes and arrays
        first_samples = first_event.r0.tel[telid].adc_samples
        n_chan, n_pixels, n_samples = first_samples.shape
        run_waveforms = np.zeros((n_events, n_pixels, n_samples),
                                 dtype=np.uint16)
        pedsub_f = np.zeros_like(run_waveforms, dtype=np.float32)
        pedsub_i = np.zeros_like(run_waveforms, dtype=np.uint16)
        ev_pedsub_f = np.zeros((n_pixels, n_samples), dtype=np.float32)
        ev_pedsub_i = np.zeros((n_pixels, n_samples), dtype=np.uint16)

        desc = "Subtracting Pedestal"
        with tqdm(total=n_events, desc=desc) as pbar:
            source = self.file_reader.read()
            for ev, event in enumerate(source):
                pbar.update(1)
                self.ped.apply(event, ev_pedsub_f)
                run_waveforms[ev] = event.r0.tel[telid].adc_samples[0]
                pedsub_f[ev, ...] = ev_pedsub_f
                pedsub_i[ev, ...] = ev_pedsub_i

        output_dir = self.output_dir
        if output_dir is None:
            output_dir = self.file_reader.output_directory

        if not os.path.exists(output_dir):
            self.log.info("[output] Creating directory: {}".format(output_dir))
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, 'pedsub.pdf')
        self.plotter.plot(run_waveforms, pedsub_f, pedsub_i, output_path)

    def finish(self):
        pass


if __name__ == '__main__':
    exe = TargetCalibStepDisplayer()
    exe.run()
