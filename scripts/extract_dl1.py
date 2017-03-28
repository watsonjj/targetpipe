from traitlets import Dict, List, Unicode
from ctapipe.core import Tool
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from targetpipe.calib.camera.waveform_cleaning import CHECMWaveformCleaner
from targetpipe.calib.camera.charge_extractors import CHECMExtractor
from targetpipe.fitting.checm import CHECMFitterSPE
from targetpipe.io.pixels import Dead
import numpy as np
from tqdm import tqdm
from os.path import join, exists
from os import makedirs


class DL1Extractor(Tool):
    name = "DL1Extractor"
    description = "Extract the dl1 information and store into a numpy file"

    adc2pe_path = Unicode('', help='Path to the numpy adc2pe '
                                   'file').tag(config=True)

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        pe='DL1Extractor.adc2pe_path',
                        t0='CHECMWaveformCleaner.t0'
                        ))
    classes = List([EventFileReaderFactory,
                    CameraR1CalibratorFactory,
                    CHECMFitterSPE,
                    CHECMWaveformCleaner
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

        self.output_dir = None

        self.adc2pe = None

        self.tack = None
        self.sec = None
        self.ns = None
        self.charge = None
        self.t0 = None
        self.baseline_start = None
        self.baseline_end = None
        self.peak_time = None
        self.fwhm = None
        self.rise_time = None

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

        self.output_dir = join(self.reader.output_directory, "extract_adc2pe")
        if not exists(self.output_dir):
            self.log.info("Creating directory: {}".format(self.output_dir))
            makedirs(self.output_dir)

        n_events = self.reader.num_events
        first_event = self.reader.get_event(0)
        n_pixels, n_samples = first_event.r0.tel[0].adc_samples[0].shape

        self.tack = np.zeros(n_events)
        self.sec = np.zeros(n_events)
        self.ns = np.zeros(n_events)
        self.charge = np.zeros((n_events, n_pixels))
        self.t0 = np.zeros(n_events)
        self.baseline_start = np.zeros((n_events, n_pixels))
        self.baseline_end = np.zeros((n_events, n_pixels))
        self.peak_time = np.zeros((n_events, n_pixels))
        self.fwhm = np.zeros((n_events, n_pixels))
        self.rise_time = np.zeros((n_events, n_pixels))

        if self.adc2pe_path:
            self.adc2pe = np.load(self.adc2pe_path)

    def start(self):
        n_events = self.reader.num_events
        first_event = self.reader.get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        n_pixels, n_samples = first_event.r0.tel[telid].adc_samples[0].shape

        # Prepare storage array
        area = np.zeros((n_events, n_pixels))
        ratio = np.ma.zeros(n_pixels)
        ratio.mask = np.zeros(ratio.shape, dtype=np.bool)
        ratio.fill_value = 0

        ind = np.indices((n_pixels, n_samples))[1]
        reversed_ind = ind[:, ::-1]

        source = self.reader.read()
        desc = "Looping through file"
        with tqdm(total=n_events, desc=desc) as pbar:
            for event in source:
                pbar.update(1)
                ev = event.count

                self.r1.calibrate(event)
                self.dl0.reduce(event)

                dl0 = np.copy(event.dl0.tel[telid].pe_samples[0])

                # Perform CHECM Waveform Cleaning
                sb_sub_wf, t0 = self.cleaner.apply(dl0)

                # Perform CHECM Charge Extraction
                peak_area, peak_height = self.extractor.extract(sb_sub_wf, t0)
                baseline_start = np.mean(sb_sub_wf[:, 0:32], axis=1)
                baseline_end = np.mean(sb_sub_wf[:, -32:], axis=1)
                peak_time = np.argmax(sb_sub_wf, axis=1)
                max_ = np.max(sb_sub_wf, axis=1)

                reversed_ = sb_sub_wf[:, ::-1]
                peak_time_i = np.ones(sb_sub_wf.shape) * peak_time[:, None]
                mask_before = np.ma.masked_less(ind, peak_time_i).mask
                mask_after = np.ma.masked_greater(reversed_ind, peak_time_i).mask
                masked_before = np.ma.masked_array(sb_sub_wf, mask_before)
                masked_after = np.ma.masked_array(reversed_, mask_after)
                half_max = max_/2
                d_l = np.diff(np.sign(half_max[:, None] - masked_after))[:, ::-1]
                d_r = np.diff(np.sign(half_max[:, None] - masked_before))
                fwhm = np.argmax(d_r, axis=1) - np.argmax(d_l, axis=1)
                _10percent = 0.1 * max_
                _90percent = 0.9 * max_
                d10 = np.diff(np.sign(_10percent[:, None] - masked_after))
                d90 = np.diff(np.sign(_90percent[:, None] - masked_after))
                rise_time = np.argmax(d10, axis=1) - np.argmax(d90, axis=1)

                charge = peak_area
                if self.adc2pe is not None:
                    charge = peak_area * self.adc2pe

                self.tack[ev] = event.meta['tack']
                self.sec[ev] = event.meta['sec']
                self.ns[ev] = event.meta['ns']
                self.charge[ev] = charge
                self.t0[ev] = t0
                self.baseline_start[ev] = baseline_start
                self.baseline_end[ev] = baseline_end
                self.peak_time[ev] = peak_time
                self.fwhm[ev] = fwhm
                self.rise_time[ev] = rise_time

    def finish(self):
        output_path = self.reader.input_path.replace("_r0.tio", "_dl1.npz")
        output_path = output_path.replace("_r1.tio", "_dl1.npz")
        np.savez(output_path,
                 tack=self.tack,
                 sec=self.sec,
                 ns=self.ns,
                 charge=self.charge,
                 t0=self.t0,
                 baseline_start=self.baseline_start,
                 baseline_end=self.baseline_end,
                 peak_time=self.peak_time,
                 fwhm=self.fwhm,
                 rise_time=self.rise_time
                 )
        self.log.info("DL1 Numpy array saved to: {}".format(output_path))

exe = DL1Extractor()
exe.run()