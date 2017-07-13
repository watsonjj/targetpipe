from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm, trange
from traitlets import Dict, List
from matplotlib import pyplot as plt, gridspec
import numpy as np
from IPython import embed

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from ctapipe.image.waveform_cleaning import WaveformCleanerFactory
from ctapipe.io.eventfilereader import EventFileReaderFactory


class EventFileLooper(Tool):
    name = "EventFileLooper"
    description = "Loop through the file and apply calibration. Intended as " \
                  "a test that the routines work, and a benchmark of speed."

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        pe='CameraR1CalibratorFactory.adc2pe_path',
                        extractor='ChargeExtractorFactory.extractor',
                        extractor_t0='ChargeExtractorFactory.t0',
                        window_width='ChargeExtractorFactory.window_width',
                        window_shift='ChargeExtractorFactory.window_shift',
                        sig_amp_cut_HG='ChargeExtractorFactory.sig_amp_cut_HG',
                        sig_amp_cut_LG='ChargeExtractorFactory.sig_amp_cut_LG',
                        lwt='ChargeExtractorFactory.lwt',
                        clip_amplitude='CameraDL1Calibrator.clip_amplitude',
                        radius='CameraDL1Calibrator.radius',
                        cleaner='WaveformCleanerFactory.cleaner',
                        cleaner_t0='WaveformCleanerFactory.t0',
                        ))
    classes = List([EventFileReaderFactory,
                    ChargeExtractorFactory,
                    CameraR1CalibratorFactory,
                    CameraDL1Calibrator,
                    WaveformCleanerFactory
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reader = None
        self.r1 = None
        self.dl0 = None
        self.dl1 = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.reader = reader_class(**kwargs)

        extractor_factory = ChargeExtractorFactory(**kwargs)
        extractor_class = extractor_factory.get_class()
        extractor = extractor_class(**kwargs)

        cleaner_factory = WaveformCleanerFactory(**kwargs)
        cleaner_class = cleaner_factory.get_class()
        cleaner = cleaner_class(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin=self.reader.origin,
                                               **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(extractor=extractor,
                                       cleaner=cleaner,
                                       **kwargs)

    def start(self):

        pix = 0

        list_f = []
        list_ax1 = []
        list_ax2 = []
        list_ax3 = []

        first_event = self.reader.get_event(0)
        n_blocks = first_event.meta['n_blocks']

        desc = "Creating figures"
        for blk in trange(n_blocks, desc=desc):
            f = plt.figure(figsize=(25, 10))
            gs = gridspec.GridSpec(2, 3)
            ax1 = f.add_subplot(gs[0, 0:2])
            ax2 = f.add_subplot(gs[1, 0:2])
            ax3 = f.add_subplot(gs[:, 2])
            f.suptitle("Block {} Row {} Column {}".format(blk, blk%8, blk//8))
            ax1.set_title("Raw")
            ax2.set_title("Pedestal Subtracted")
            ax1.set_xlabel("Blockphase + Sample Number")
            ax3.set_xlabel("Blockphase")
            ax3.set_ylabel("Raw Waveform Average")
            list_f.append(f)
            list_ax1.append(ax1)
            list_ax2.append(ax2)
            list_ax3.append(ax3)

        n_events = self.reader.num_events
        source = self.reader.read()
        desc = "Looping through file"
        for event in tqdm(source, desc=desc, total=n_events):
            r = event.r0.tel[0].row
            c = event.r0.tel[0].column
            blk = c * 8 + r
            bph = event.r0.tel[0].blockphase



            ev = event.count
            self.r1.calibrate(event)
            self.dl0.reduce(event)
            # self.dl1.calibrate(event)

            r0 = event.r0.tel[0].adc_samples[0]
            r1 = event.r1.tel[0].pe_samples[0]

            m_pix_raw = r0[pix].mean()
            m_sub = r1.mean()
            color = 'r' if m_sub < -0.75 else 'b'

            x = np.linspace(bph[pix], bph[pix]+r0.shape[1], r0.shape[1])
            list_ax1[blk[pix]].plot(x, r0[pix], color=color, linewidth=0.1)
            list_ax2[blk[pix]].plot(x, r1[pix], color=color, linewidth=0.1)
            list_ax3[blk[pix]].plot(bph[pix], m_pix_raw, color=color, marker='x')

        output_path = "/Users/Jason/Downloads/pedestal_wfs.pdf"
        with PdfPages(output_path) as pdf:
            desc = "Saving pages to pdf"
            for blk in trange(n_blocks, desc=desc):
                pdf.savefig(list_f[blk])

    def finish(self):
        pass


if __name__ == '__main__':
    exe = EventFileLooper()
    exe.run()
