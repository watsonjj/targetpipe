#!python
"""
Create a animated gif of an waveforms, similar to those produced in libCHEC for
the inauguration press release.
"""

from os import makedirs
from os.path import join, exists

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm
from traitlets import Dict, List, Int

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.core import Tool, Component
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from ctapipe.image.waveform_cleaning import WaveformCleanerFactory
from ctapipe.instrument import CameraGeometry
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.visualization import CameraDisplay
from targetpipe.fitting.checm import CHECMFitterSPE
from targetpipe.io.pixels import Dead


class Animator(Component):
    name = 'Animator'

    start = Int(40, help='Time to start gif').tag(config=True)
    end = Int(8, help='Time to end gif').tag(config=True)
    p1 = Int(0, help='Pixel 1').tag(config=True)
    p2 = Int(0, help='Pixel 2').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, parent=tool, **kwargs)

        self.fig = plt.figure(figsize=(24, 10))
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax2 = self.fig.add_subplot(2, 2, 3)
        self.camera = self.fig.add_subplot(1, 2, 2)

    def plot(self, waveforms, geom, event_id, output_dir):
        camera = CameraDisplay(geom, ax=self.camera, image=np.zeros(2048),
                               cmap='viridis')
        camera.add_colorbar()
        max_ = np.percentile(waveforms[:, self.start:self.end].max(), 60)
        camera.set_limits_minmax(0, max_)

        self.ax1.plot(waveforms[self.p1, :])
        self.ax2.plot(waveforms[self.p2, :])

        self.fig.suptitle("Event {}".format(event_id))
        self.ax1.set_title("Pixel: {}".format(self.p1))
        self.ax1.set_xlabel("Time (ns)")
        self.ax1.set_ylabel("Amplitude (p.e.)")
        self.ax2.set_title("Pixel: {}".format(self.p2))
        self.ax2.set_xlabel("Time (ns)")
        self.ax2.set_ylabel("Amplitude (p.e.)")
        camera.colorbar.set_label("Amplitude (p.e.)")

        line1, = self.ax1.plot([0, 0], self.ax1.get_ylim(), color='r', alpha=1)
        line2, = self.ax2.plot([0, 0], self.ax2.get_ylim(), color='r', alpha=1)

        self.camera.annotate(
            "Pixel: {}".format(self.p1),
            xy=(geom.pix_x.value[self.p1], geom.pix_y.value[self.p1]),
            xycoords='data', xytext=(0.05, 0.98),
            textcoords='axes fraction',
            arrowprops=dict(facecolor='red', width=2, alpha=0.4),
            horizontalalignment='left', verticalalignment='top')
        self.camera.annotate(
            "Pixel: {}".format(self.p2),
            xy=(geom.pix_x.value[self.p2], geom.pix_y.value[self.p2]),
            xycoords='data', xytext=(0.05, 0.94),
            textcoords='axes fraction',
            arrowprops=dict(facecolor='orange', width=2, alpha=0.4),
            horizontalalignment='left', verticalalignment='top')

        # Create animation
        div = 5
        increment = 1/div
        n_frames = int((self.end - self.start)/increment)
        interval = int(500*increment)

        # Prepare Output
        output_path = join(output_dir, "animation_e{}.gif".format(event_id))
        if not exists(output_dir):
            self.log.info("Creating directory: {}".format(output_dir))
            makedirs(output_dir)
        self.log.info("Output: {}".format(output_path))

        with tqdm(total=n_frames, desc="Creating animation") as pbar:
            def animate(i):
                pbar.update(1)
                t = self.start + (i / div)
                camera.image = waveforms[:, int(t)]
                line1.set_xdata(t)
                line2.set_xdata(t)

            anim = animation.FuncAnimation(self.fig, animate, frames=n_frames,
                                           interval=interval)
            anim.save(output_path, writer='imagemagick')

        self.log.info("Created animation: {}".format(output_path))


class EventAnimationCreator(Tool):
    name = "EventAnimationCreator"
    description = "Create an animation of the camera image through timeslices"

    req_event = Int(0, help='Event to plot').tag(config=True)

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        pe='CameraR1CalibratorFactory.adc2pe_path',
                        cleaner='WaveformCleanerFactory.cleaner',
                        e='EventAnimationCreator.req_event',
                        start='Animator.start',
                        end='Animator.end',
                        p1='Animator.p1',
                        p2='Animator.p2'
                        ))
    classes = List([EventFileReaderFactory,
                    CameraR1CalibratorFactory,
                    WaveformCleanerFactory,
                    CHECMFitterSPE,
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

        cleaner_factory = WaveformCleanerFactory(**kwargs)
        cleaner_class = cleaner_factory.get_class()
        self.cleaner = cleaner_class(**kwargs)

        extractor_factory = ChargeExtractorFactory(**kwargs)
        extractor_class = extractor_factory.get_class()
        self.extractor = extractor_class(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(extractor=self.extractor,
                                       cleaner=self.cleaner,
                                       **kwargs)

        self.fitter = CHECMFitterSPE(**kwargs)
        self.dead = Dead()

        self.animator = Animator(**kwargs)

    def start(self):
        event = self.reader.get_event(self.req_event)
        telid = list(event.r0.tels_with_data)[0]
        geom = CameraGeometry.guess(*event.inst.pixel_pos[0],
                                    event.inst.optical_foclen[0])

        self.r1.calibrate(event)
        self.dl0.reduce(event)
        self.dl1.calibrate(event)

        cleaned = event.dl1.tel[telid].cleaned[0]

        output_dir = self.reader.output_directory
        self.animator.plot(cleaned, geom, self.req_event, output_dir)

    def finish(self):
        pass

exe = EventAnimationCreator()
exe.run()
