from os import makedirs
from os.path import join, exists

import numpy as np
from tqdm import tqdm
from traitlets import Dict, List

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from ctapipe.image.waveform_cleaning import WaveformCleanerFactory
from ctapipe.io.eventfilereader import EventFileReaderFactory
from targetpipe.io.pixels import Dead

from IPython import embed


class DL1Extractor(Tool):
    name = "DL1Extractor"
    description = "Extract the dl1 information and store into a numpy file"

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        pe='CameraR1CalibratorFactory.adc2pe_path',
                        cleaner='WaveformCleanerFactory.cleaner',
                        extractor='ChargeExtractorFactory.extractor',
                        extractor_t0='ChargeExtractorFactory.t0',
                        window_width='ChargeExtractorFactory.window_width',
                        window_shift='ChargeExtractorFactory.window_shift',
                        radius='CameraDL1Calibrator.radius',
                        ))

    classes = List([EventFileReaderFactory,
                    CameraR1CalibratorFactory,
                    WaveformCleanerFactory,
                    ChargeExtractorFactory,
                    CameraDL1Calibrator
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reader = None
        self.r1 = None
        self.dl0 = None
        self.dl1 = None
        self.cleaner = None
        self.extractor = None

        self.dead = None

        self.output_dir = None

        self.tack = None
        self.sec = None
        self.ns = None
        self.fci = None
        self.charge = None
        self.t0 = None
        self.baseline_mean_start = None
        self.baseline_mean_end = None
        self.baseline_mean_full = None
        self.baseline_rms_start = None
        self.baseline_rms_end = None
        self.baseline_rms_full = None
        self.peak_time = None
        self.fwhm = None
        self.rise_time = None
        self.n_saturated = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        cleaner_factory = WaveformCleanerFactory(**kwargs)
        cleaner_class = cleaner_factory.get_class()
        self.cleaner = cleaner_class(**kwargs)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.reader = reader_class(**kwargs)

        extractor_factory = ChargeExtractorFactory(**kwargs)
        extractor_class = extractor_factory.get_class()
        self.extractor = extractor_class(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin=self.reader.origin,
                                               **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(extractor=self.extractor,
                                       cleaner=self.cleaner,
                                       **kwargs)

        self.dead = Dead()

        self.output_dir = join(self.reader.output_directory, "extract_adc2pe")
        if not exists(self.output_dir):
            self.log.info("Creating directory: {}".format(self.output_dir))
            makedirs(self.output_dir)

        n_events = self.reader.num_events
        first_event = self.reader.get_event(0)
        n_pixels = first_event.inst.num_pixels[0]
        n_samples = first_event.r0.tel[0].num_samples

        self.tack = np.zeros(n_events)
        self.sec = np.zeros(n_events)
        self.ns = np.zeros(n_events)
        self.fci = np.zeros((n_events, n_pixels))
        self.charge = np.zeros((n_events, n_pixels))
        self.t0 = np.zeros((n_events, n_pixels))
        self.baseline_mean_start = np.zeros((n_events, n_pixels))
        self.baseline_mean_end = np.zeros((n_events, n_pixels))
        self.baseline_mean_full = np.zeros((n_events, n_pixels))
        self.baseline_rms_start = np.zeros((n_events, n_pixels))
        self.baseline_rms_end = np.zeros((n_events, n_pixels))
        self.baseline_rms_full = np.zeros((n_events, n_pixels))
        self.peak_time = np.zeros((n_events, n_pixels))
        self.fwhm = np.zeros((n_events, n_pixels))
        self.rise_time = np.zeros((n_events, n_pixels))
        self.n_saturated = np.zeros((n_events, n_pixels))
        #Justus:
        self.n_1pe = np.zeros((n_events, n_pixels))
        self.peak_height = np.zeros((n_events, n_pixels))

    def start(self):
        n_events = self.reader.num_events
        first_event = self.reader.get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        n_pixels = first_event.inst.num_pixels[0]
        n_samples = first_event.r0.tel[0].num_samples

        ind = np.indices((n_pixels, n_samples))[1]
        r_ind = ind[:, ::-1]

        source = self.reader.read()
        desc = "Looping through file"
        with tqdm(total=n_events, desc=desc) as pbar:
            for event in source:
                pbar.update(1)
                ev = event.count

                self.r1.calibrate(event)
                self.dl0.reduce(event)
                self.dl1.calibrate(event)

                peak_area = event.dl1.tel[telid].image
                t0 = event.dl1.tel[telid].peakpos[0]
                dl0 = event.dl0.tel[telid].pe_samples[0]
                peak_time = np.argmax(dl0, axis=1)
                # cleaned = event.dl1.tel[telid].cleaned[0]

                # sb_sub_wf = np.ma.masked_where(sb_sub_wf < -200, sb_sub_wf)

                baseline_mean_start = np.mean(dl0[:, 0:32], axis=1)
                baseline_mean_end = np.mean(dl0[:, -32:], axis=1)
                baseline_mean_full = np.mean(dl0[:, 10:-10], axis=1)
                baseline_rms_start = np.std(dl0[:, 0:32], axis=1)
                baseline_rms_end = np.std(dl0[:, -32:], axis=1)
                baseline_rms_full = np.std(dl0[:, 10:-10], axis=1)

                max_ = np.max(dl0, axis=1)
                reversed_ = dl0[:, ::-1]
                peak_time_i = np.ones(dl0.shape) * peak_time[:, None]
                mask_before = np.ma.masked_less(ind, peak_time_i).mask
                mask_after = np.ma.masked_greater(r_ind, peak_time_i).mask
                masked_bef = np.ma.masked_array(dl0, mask_before)
                masked_aft = np.ma.masked_array(reversed_, mask_after)
                half_max = max_/2
                d_l = np.diff(np.sign(half_max[:, None] - masked_aft))
                d_r = np.diff(np.sign(half_max[:, None] - masked_bef))
                t_l = r_ind[0, np.argmax(d_l, axis=1) + 1]
                t_r = ind[0, np.argmax(d_r, axis=1) + 1]
                fwhm = t_r - t_l
                _10percent = 0.1 * max_
                _90percent = 0.9 * max_
                d10 = np.diff(np.sign(_10percent[:, None] - masked_aft))
                d90 = np.diff(np.sign(_90percent[:, None] - masked_aft))
                t10 = r_ind[0, np.argmax(d10, axis=1) + 1]
                t90 = r_ind[0, np.argmax(d90, axis=1) + 1]
                rise_time = t90 - t10

                self.tack[ev] = event.meta['tack']
                self.sec[ev] = event.meta['sec']
                self.ns[ev] = event.meta['ns']
                self.fci[ev] = event.r0.tel[0].first_cell_ids
                self.charge[ev] = peak_area
                self.t0[ev] = t0
                self.baseline_mean_start[ev] = baseline_mean_start
                self.baseline_mean_end[ev] = baseline_mean_end
                self.baseline_mean_full[ev] = baseline_mean_full
                self.baseline_rms_start[ev] = baseline_rms_start
                self.baseline_rms_end[ev] = baseline_rms_end
                self.baseline_rms_full[ev] = baseline_rms_full
                self.peak_time[ev] = peak_time
                if np.ma.is_masked(dl0):
                    self.n_saturated[ev] = np.sum(dl0.mask, axis=1)
                # Justus:
                self.n_1pe[ev] = (dl0 >= 0.6).sum(axis=1)
                self.peak_height[ev] = dl0.max(1)
                self.fwhm[ev] = fwhm
                self.rise_time[ev] = rise_time

                #from IPython import embed
                #embed()

                #print(self.tack[ev]-self.tack[ev-1])

    def finish(self):
        output_path = self.reader.input_path.replace("_r0.tio", "_dl1.npz")
        output_path = output_path.replace("_r1.tio", "_dl1.npz")
        np.savez(output_path,
                 tack=self.tack,
                 sec=self.sec,
                 ns=self.ns,
                 fci=self.fci,
                 charge=self.charge,
                 t0=self.t0,
                 baseline_mean_start=self.baseline_mean_start,
                 baseline_mean_end=self.baseline_mean_end,
                 baseline_mean_full=self.baseline_mean_full,
                 baseline_rms_start=self.baseline_rms_start,
                 baseline_rms_end=self.baseline_rms_end,
                 baseline_rms_full=self.baseline_rms_full,
                 peak_time=self.peak_time,
                 n_saturated=self.n_saturated,
                 # Justus:
                 n_1pe = self.n_1pe,
                 peak_height = self.peak_height,
                 fwhm=self.fwhm,
                 rise_time=self.rise_time
                 )
        self.log.info("DL1 Numpy array saved to: {}".format(output_path))

exe = DL1Extractor()
exe.run()
