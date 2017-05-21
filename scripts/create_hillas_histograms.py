#!python
"""
Create a animated gif of an waveforms, similar to those produced in libCHEC for
the inauguration press release.
"""

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from traitlets import Dict, List

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.core import Tool
from ctapipe.image import tailcuts_clean
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from ctapipe.image.hillas import HillasParameterizationError, \
    hillas_parameters
from ctapipe.image.waveform_cleaning import CHECMWaveformCleanerLocal
from ctapipe.instrument import CameraGeometry
from ctapipe.io.eventfilereader import EventFileReaderFactory
from targetpipe.fitting.checm import CHECMFitterSPE
from targetpipe.io.pixels import Dead


class HillasExtractor(Tool):
    name = "HillasExtractor"
    description = "Extract the hillas parameters and store them in various " \
                  "forms."

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        pe='CameraR1CalibratorFactory.adc2pe_path',
                        ))
    classes = List([EventFileReaderFactory,
                    CameraR1CalibratorFactory,
                    CHECMFitterSPE,
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

        self.n_events = None

        self.time = None
        self.size = None
        self.cen_x = None
        self.cen_y = None
        self.length = None
        self.width = None
        self.r = None
        self.phi = None
        self.psi = None
        self.miss = None
        self.skewness = None
        self.kurtosis = None

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

        self.fitter = CHECMFitterSPE(**kwargs)
        self.dead = Dead()

        self.n_events = self.reader.num_events

        self.time = np.ma.zeros(self.n_events)
        self.size = np.ma.zeros(self.n_events)
        self.cen_x = np.ma.zeros(self.n_events)
        self.cen_y = np.ma.zeros(self.n_events)
        self.length = np.ma.zeros(self.n_events)
        self.width = np.ma.zeros(self.n_events)
        self.r = np.ma.zeros(self.n_events)
        self.phi = np.ma.zeros(self.n_events)
        self.psi = np.ma.zeros(self.n_events)
        self.miss = np.ma.zeros(self.n_events)
        self.skewness = np.ma.zeros(self.n_events)
        self.kurtosis = np.ma.zeros(self.n_events)

    def start(self):
        first_event = self.reader.get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        pos = first_event.inst.pixel_pos[telid]
        foclen = first_event.inst.optical_foclen[telid]
        geom = CameraGeometry.guess(*pos, foclen)

        mask = np.zeros(self.n_events, dtype=bool)

        desc = "Extracting image slices from file"
        source = self.reader.read()
        for event in tqdm(source, total=self.n_events, desc=desc):
            for telid in event.r0.tels_with_data:
                ev = event.count

                self.r1.calibrate(event)
                self.dl0.reduce(event)
                self.dl1.calibrate(event)

                image = event.dl1.tel[telid].image[0]

                # Cleaning
                tc = tailcuts_clean(geom, image, 20, 10)
                if not tc.any():
                    mask[ev] = True
                    continue
                cleaned_dl1 = np.ma.masked_array(image, mask=~tc)

                try:
                    hillas = hillas_parameters(*pos, cleaned_dl1)
                except HillasParameterizationError:
                    mask[ev] = True
                    print('HillasParameterizationError')
                    continue

                self.time[ev] = event.trig.gps_time.value
                self.size[ev] = hillas.size
                self.cen_x[ev] = hillas.cen_x.value
                self.cen_y[ev] = hillas.cen_y.value
                self.length[ev] = hillas.length.value
                self.width[ev] = hillas.width.value
                self.r[ev] = hillas.r.value
                self.phi[ev] = hillas.phi.value
                self.psi[ev] = hillas.psi.value
                self.miss[ev] = hillas.miss.value
                self.skewness[ev] = hillas.skewness
                self.kurtosis[ev] = hillas.kurtosis

                if np.isnan(self.width[ev]):
                    mask[ev] = True

        self.time.mask = mask
        self.size.mask = mask
        self.cen_x.mask = mask
        self.cen_y.mask = mask
        self.length.mask = mask
        self.width.mask = mask
        self.r.mask = mask
        self.phi.mask = mask
        self.psi.mask = mask
        self.miss.mask = mask
        self.skewness.mask = mask
        self.kurtosis.mask = mask

    def finish(self):
        fig = plt.figure(figsize=(24, 10))
        ax1 = fig.add_subplot(2, 3, 1)
        ax2 = fig.add_subplot(2, 3, 2)
        ax3 = fig.add_subplot(2, 3, 3)
        ax4 = fig.add_subplot(2, 3, 4)
        ax5 = fig.add_subplot(2, 3, 5)
        ax6 = fig.add_subplot(2, 3, 6)

        ax1.set_title("Width")
        ax2.set_title("Length")
        ax3.set_title("Size")
        ax4.set_title("Phi")
        ax5.set_title("Miss")
        ax6.set_title("R")

        ax1.hist(self.width.compressed(), bins=60)
        ax2.hist(self.length.compressed(), bins=60)
        ax3.hist(self.size.compressed(), bins=60)
        ax4.hist(self.phi.compressed(), bins=60)
        ax5.hist(self.miss.compressed(), bins=60)
        ax6.hist(self.r.compressed(), bins=60)

        fig.savefig("/Users/Jason/Downloads/hillas.png")
        np.savez("/Users/Jason/Downloads/hillas.npz",
                 time=self.time.compressed(),
                 size=self.size.compressed(),
                 cen_x=self.cen_x.compressed(),
                 cen_y=self.cen_y.compressed(),
                 length=self.length.compressed(),
                 width=self.width.compressed(),
                 r=self.r.compressed(),
                 phi=self.phi.compressed(),
                 psi=self.psi.compressed(),
                 miss=self.miss.compressed(),
                 skewness=self.skewness.compressed(),
                 kurtosis=self.kurtosis.compressed())

        with open("/Users/Jason/Downloads/hillas.csv", 'w') as f:
            f.write("time,size,cen_x,cen_y,length,width,r,phi,psi,miss,"
                    "skewness,kurtosis\n")
            for ev in range(self.time.compressed().size):
                f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n"
                        .format(self.time.compressed()[ev],
                                self.size.compressed()[ev],
                                self.cen_x.compressed()[ev],
                                self.cen_y.compressed()[ev],
                                self.length.compressed()[ev],
                                self.width.compressed()[ev],
                                self.r.compressed()[ev],
                                self.phi.compressed()[ev],
                                self.psi.compressed()[ev],
                                self.miss.compressed()[ev],
                                self.skewness.compressed()[ev],
                                self.kurtosis.compressed()[ev]))


exe = HillasExtractor()
exe.run()
