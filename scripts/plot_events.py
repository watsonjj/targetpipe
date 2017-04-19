from traitlets import Dict, List
from ctapipe.core import Tool
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.charge_extractors import ChargeExtractorFactory
from ctapipe.calib.camera.waveform_cleaning import WaveformCleanerFactory
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from os.path import join, exists
from os import makedirs
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
import numpy as np
from ctapipe.image import tailcuts_clean
from ctapipe.image.hillas import HillasParameterizationError, hillas_parameters_4


class EventFileLooper(Tool):
    name = "EventFileLooper"
    description = "Loop through the file and apply calibration. Intended as " \
                  "a test that the routines work, and a benchmark of speed."

    aliases = Dict(dict(f='EventFileReaderFactory.input_path',
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

        # Get first event information
        first_event = self.reader.get_event(0)
        r0 = first_event.r0.tel[0].adc_samples[0]
        n_pixels, n_samples = r0.shape
        pos = first_event.inst.pixel_pos[0]
        foclen = first_event.inst.optical_foclen[0]
        geom = CameraGeometry.guess(*pos, foclen)

        # Setup Output
        output_dir = self.reader.output_directory
        title = self.reader.filename
        title = title[:title.find("_")]
        # Prepare Output
        if not exists(output_dir):
            self.log.info("Creating directory: {}".format(output_dir))
            makedirs(output_dir)
        output_path = join(output_dir, title + "_events.pdf")

        # Setup plot
        fig = plt.figure(figsize=(10, 10))
        ax_camera = fig.add_subplot(1, 1, 1)
        fig.patch.set_visible(False)
        ax_camera.axis('off')
        camera = CameraDisplay(geom, ax=ax_camera, image=np.zeros(2048),
                               cmap='viridis')
        camera.add_colorbar()
        cb = camera.colorbar
        camera.colorbar.set_label("Amplitude (p.e.)")
        fig.suptitle(title)

        source = self.reader.read()
        desc = "Looping through file"
        with PdfPages(output_path) as pdf:
            for event in tqdm(source, desc=desc):
                ev = event.count
                self.r1.calibrate(event)
                self.dl0.reduce(event)
                self.dl1.calibrate(event)
                for t in event.r0.tels_with_data:
                    dl1 = event.dl1.tel[t].image[0]

                    # Cleaning
                    tc = tailcuts_clean(geom, dl1, 20, 10)
                    # dilate(geom, tc)
                    # dilate(geom, tc)
                    cleaned_tc = np.ma.masked_array(dl1, mask=~tc)

                    try:
                        # hillas = hillas_parameters(*pos, cleaned_tc)
                        hillas = hillas_parameters_4(*pos, cleaned_tc)
                    except HillasParameterizationError:
                        continue

                    ax_camera.cla()
                    camera = CameraDisplay(geom, ax=ax_camera,
                                           image=np.zeros(2048),
                                           cmap='viridis')
                    camera.colorbar = cb
                    camera.image = dl1
                    camera.overlay_moments(hillas, color='red')
                    camera.update(True)
                    ax_camera.set_title("Event: {}".format(ev))
                    ax_camera.axis('off')

                    pdf.savefig(fig)

        self.log.info("Created images: {}".format(output_path))

    def finish(self):
        pass


if __name__ == '__main__':
    exe = EventFileLooper()
    exe.run()
