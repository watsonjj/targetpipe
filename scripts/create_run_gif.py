from os import makedirs
from os.path import join, exists

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm
from traitlets import Dict, List, Unicode
import pandas as pd

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.core import Tool, Component
from ctapipe.image import tailcuts_clean
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from ctapipe.image.waveform_cleaning import CHECMWaveformCleanerLocal
from ctapipe.instrument import CameraGeometry
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.visualization import CameraDisplay
from targetpipe.fitting.chec import CHECMSPEFitter
from targetpipe.io.pixels import Dead


class Animator(Component):
    name = 'Animator'

    description = Unicode("", help="Description for the run "
                                   "file").tag(config=True)

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, parent=tool, **kwargs)

        self.fig = plt.figure(figsize=(10, 10))
        self.ax_camera = self.fig.add_subplot(1, 1, 1)
        self.fig.patch.set_visible(False)
        self.ax_camera.axis('off')

    def plot(self, df, n_frames, geom, output_path, title):
        camera = CameraDisplay(geom, ax=self.ax_camera, image=np.zeros(2048),
                               cmap='viridis')
        camera.add_colorbar()
        camera.colorbar.set_label("Amplitude (p.e.)")
        self.fig.suptitle(title + " - " + self.description)

        # Create animation
        interval = 25  # Fast
        # interval = 100  # Slow

        def animation_generator():
            for index, row in df.iterrows():
                event_id = row['event_id']
                images = row['images']
                tc = row['tc']

                # max_ = np.percentile(event.max(), 60)
                # camera.image = image
                # camera.set_limits_minmax(min_, max_)

                tc_2d = np.ones(images.shape, dtype=np.bool) * tc[None, :]
                cleaned_events = np.ma.masked_array(images, mask=~tc_2d)
                max_ = cleaned_events.max()  # np.percentile(dl1, 99.9)
                if max_ < 6:
                    max_ = 6
                min_ = np.percentile(images, 0.1)

                camera.set_limits_minmax(min_, max_)
                self.ax_camera.set_title("Event: {}".format(event_id))
                for s in images:
                    camera.image = s
                    yield
        source = animation_generator()

        self.log.info("Output: {}".format(output_path))
        with tqdm(total=n_frames, desc="Creating animation") as pbar:
            def animate(_):
                pbar.update(1)
                next(source)

            anim = animation.FuncAnimation(self.fig, animate,
                                           frames=n_frames-1,
                                           interval=interval)
            anim.save(output_path)

        self.log.info("Created animation: {}".format(output_path))


class EventAnimationCreator(Tool):
    name = "EventAnimationCreator"
    description = "Create an animation for every event in a run"

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        pe='CameraR1CalibratorFactory.pe_path',
                        desc='Animator.description',
                        ))
    classes = List([EventFileReaderFactory,
                    CameraR1CalibratorFactory,
                    CHECMSPEFitter,
                    Animator
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reader = None
        self.r1 = None
        self.dl0 = None
        self.cleaner = None
        self.extractor = None
        self.dl1 = None

        self.fitter = None
        self.dead = None

        self.adc2pe = None

        self.animator = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.reader = reader_class(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin=self.reader.origin,
                                               **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)

        self.cleaner = CHECMWaveformCleanerLocal(**kwargs)

        extractor_factory = ChargeExtractorFactory(**kwargs)
        extractor_class = extractor_factory.get_class()
        self.extractor = extractor_class(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(extractor=self.extractor,
                                       cleaner=self.cleaner,
                                       **kwargs)

        self.fitter = CHECMSPEFitter(**kwargs)
        self.dead = Dead()

        self.animator = Animator(**kwargs)

    def start(self):
        df_list = []
        n_frames = 0

        first_event = self.reader.get_event(0)
        n_samples = first_event.r0.tel[0].num_samples
        pos = first_event.inst.pixel_pos[0]
        foclen = first_event.inst.optical_foclen[0]
        geom = CameraGeometry.guess(*pos, foclen)

        desc = "Extracting image slices from file"
        n_events = self.reader.num_events
        source = self.reader.read()
        for event in tqdm(source, total=n_events, desc=desc):
            event_id = event.r0.event_id

            self.r1.calibrate(event)
            self.dl0.reduce(event)
            self.dl1.calibrate(event)

            image = event.dl1.tel[0].image[0]
            cleaned = event.dl1.tel[0].cleaned[0]

            # Cleaning
            tc = tailcuts_clean(geom, image, 7, 3)
            empty = np.zeros(cleaned.shape, dtype=bool)
            cleaned_tc_mask = np.ma.mask_or(empty, ~tc[:, None])
            cleaned_dl1 = np.ma.masked_array(cleaned, mask=cleaned_tc_mask)

            # Find start and end of movie for event
            sum_wf = np.sum(cleaned_dl1, axis=0)
            sum_wf_t = np.arange(sum_wf.size)
            max_ = np.max(sum_wf)
            tmax = np.argmax(sum_wf)
            before = sum_wf[:tmax+1][::-1]
            before_t = sum_wf_t[:tmax+1][::-1]
            after = sum_wf[tmax:]
            after_t = sum_wf_t[tmax:]
            limit = 0.1 * max_
            # if limit < 2:
            #     limit = 2
            try:
                start = before_t[before <= limit][0] - 2
                end = after_t[after <= limit][0] + 5
            except IndexError:
                self.log.warning("No image for event id {}".format(event_id))
                continue
            if start < 0:
                start = 0
            if end >= n_samples:
                end = n_samples-1

            s = []
            for t in range(start, end):
                s.append(cleaned[:, t])
                n_frames += 1
            df_list.append(dict(event_id=event_id, images=np.array(s), tc=tc))

        df = pd.DataFrame(df_list)

        output_dir = self.reader.output_directory
        title = self.reader.filename
        title = title[:title.find("_")]
        # Prepare Output
        if not exists(output_dir):
            self.log.info("Creating directory: {}".format(output_dir))
            makedirs(output_dir)
        output_path = join(output_dir, title+"_animation.mp4")

        self.animator.plot(df, n_frames, geom, output_path, title)

    def finish(self):
        pass

exe = EventAnimationCreator()
exe.run()
