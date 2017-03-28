#!python
"""
Create a animated gif of an waveforms, similar to those produced in libCHEC for
the inauguration press release.
"""

from traitlets import Dict, List, Unicode, Int
from ctapipe.core import Tool, Component
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.io import CameraGeometry
from ctapipe.visualization import CameraDisplay
from targetpipe.calib.camera.waveform_cleaning import CHECMWaveformCleaner
from targetpipe.calib.camera.charge_extractors import CHECMExtractor
from targetpipe.fitting.checm import CHECMFitterSPE
from targetpipe.io.pixels import Dead
import numpy as np
from tqdm import tqdm
from os.path import join, exists
from os import makedirs
from matplotlib import pyplot as plt
from matplotlib import animation


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
        max_ = np.percentile(waveforms[:, self.start:self.end].max()*0.5, 60)
        camera.set_limits_minmax(0, max_)

        self.ax1.plot(waveforms[self.p1, :])
        self.ax2.plot(waveforms[self.p2, :])

        self.fig.suptitle("Event {}".format(event_id))
        self.ax1.set_title("Pixel: {}".format(self.p1))
        self.ax1.set_xlabel("Time (ns)")
        self.ax1.set_ylabel("Amplitude (Calibrated ADC)")
        self.ax2.set_title("Pixel: {}".format(self.p2))
        self.ax2.set_xlabel("Time (ns)")
        self.ax2.set_ylabel("Amplitude (Calibrated ADC)")

        line1, = self.ax1.plot([0, 0], self.ax1.get_ylim(), color='r', alpha=1)
        line2, = self.ax2.plot([0, 0], self.ax2.get_ylim(), color='r', alpha=1)

        self.camera.annotate("Pixel: {}".format(self.p1),
                          xy=(geom.pix_x.value[self.p1],
                          geom.pix_y.value[self.p1]),
                          xycoords='data', xytext=(0.05, 0.98),
                          textcoords='axes fraction',
                          arrowprops=dict(facecolor='red', width=2, alpha=0.4),
                          horizontalalignment='left', verticalalignment='top')
        self.camera.annotate("Pixel: {}".format(self.p2),
                          xy=(geom.pix_x.value[self.p2],
                          geom.pix_y.value[self.p2]),
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


class EventAnimationCreator(Tool):
    name = "EventAnimationCreator"
    description = "Create an animation of the camera image through timeslices"

    adc2pe_path = Unicode('', help='Path to the numpy adc2pe '
                                   'file').tag(config=True)
    req_event = Int(0, help='Event to plot').tag(config=True)

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        # pe='DL1Extractor.adc2pe_path',
                        t0='CHECMWaveformCleaner.t0',
                        e='EventAnimationCreator.req_event',
                        start='Animator.start',
                        end='Animator.end',
                        p1='Animator.p1',
                        p2='Animator.p2'
                        ))
    classes = List([EventFileReaderFactory,
                    CameraR1CalibratorFactory,
                    CHECMWaveformCleaner,
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

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.cleaner = CHECMWaveformCleaner(**kwargs)
        self.extractor = CHECMExtractor(**kwargs)
        self.fitter = CHECMFitterSPE(**kwargs)
        self.dead = Dead()

        if self.adc2pe_path:
            self.adc2pe = np.load(self.adc2pe_path)

        self.animator = Animator(**kwargs)

    def start(self):
        event = self.reader.get_event(self.req_event)
        telid = list(event.r0.tels_with_data)[0]
        geom = CameraGeometry.guess(*event.inst.pixel_pos[0],
                                    event.inst.optical_foclen[0])

        self.r1.calibrate(event)
        self.dl0.reduce(event)

        dl0 = np.copy(event.dl0.tel[telid].pe_samples[0])

        # Perform CHECM Waveform Cleaning
        sb_sub_wf, t0 = self.cleaner.apply(dl0)

        output_dir = self.reader.output_directory
        self.animator.plot(sb_sub_wf, geom, self.req_event, output_dir)

    def finish(self):
        pass

exe = EventAnimationCreator()
exe.run()
