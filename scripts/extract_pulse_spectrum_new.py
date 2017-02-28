from traitlets import Dict, List
from ctapipe.core import Tool, Component
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from os import makedirs
from os.path import join, exists
from targetpipe.calib.camera.waveform_cleaning import CHECMWaveformCleaner
from targetpipe.calib.camera.charge_extractors import CHECMExtractor


class EventFileLooper(Tool):
    name = "EventFileLooper"
    description = "Loop through the file and apply calibration. Intended as " \
                  "a test that the routines work, and a benchmark of speed."

    aliases = Dict(dict(f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        # extractor='ChargeExtractorFactory.extractor',
                        # window_width='ChargeExtractorFactory.window_width',
                        # window_start='ChargeExtractorFactory.window_start',
                        # window_shift='ChargeExtractorFactory.window_shift',
                        # sig_amp_cut_HG='ChargeExtractorFactory.sig_amp_cut_HG',
                        # sig_amp_cut_LG='ChargeExtractorFactory.sig_amp_cut_LG',
                        # lwt='ChargeExtractorFactory.lwt',
                        # clip_amplitude='CameraDL1Calibrator.clip_amplitude',
                        # radius='CameraDL1Calibrator.radius',
                        ))
    classes = List([EventFileReaderFactory,
                    # ChargeExtractorFactory,
                    CameraR1CalibratorFactory,
                    # CameraDL1Calibrator,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_reader = None
        self.r1 = None
        self.dl0 = None
        self.dl1 = None

        self.cleaner = None
        self.extractor = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.file_reader = reader_class(**kwargs)

        # extractor_factory = ChargeExtractorFactory(**kwargs)
        # extractor_class = extractor_factory.get_class()
        # self.extractor = extractor_class(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin=self.file_reader.origin,
                                               **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        # self.dl1 = CameraDL1Calibrator(extractor=self.extractor, **kwargs)

        self.cleaner = CHECMWaveformCleaner(**kwargs)
        self.extractor = CHECMExtractor(**kwargs)

    def start(self):

        # Look at plots of first event
        event_index = 100
        event0 = self.file_reader.get_event(event_index)
        self.r1.calibrate(event0)
        self.dl0.reduce(event0)
        telid = list(event0.r0.tels_with_data)[0]
        dl0 = np.copy(event0.dl0.tel[telid].pe_samples[0])
        n_events = self.file_reader.num_events
        n_pixels, n_samples = dl0.shape

        # Perform CHECM Waveform Cleaning
        sb_sub_wf, t0 = self.cleaner.apply(dl0)
        baseline_sub = self.cleaner.stages['1: baseline_sub']
        avg_wf = self.cleaner.stages['2: avg_wf']
        fit_wf = self.cleaner.stages['3: fit_wf']
        pw_l = self.cleaner.pw_l
        pw_r = self.cleaner.pw_r
        no_pulse = self.cleaner.stages['4: no_pulse']
        smooth_baseline = self.cleaner.stages['5: smooth_baseline']
        smooth_wf = self.cleaner.stages['6: smooth_wf']

        # Perform CHECM Charge Extraction
        self.extractor.extract(sb_sub_wf, t0)
        iw_l = self.extractor.iw_l
        iw_r = self.extractor.iw_r

        # Prepare plots
        n_plots = 9
        baseline_fig = plt.figure(figsize=(13, 13))
        baseline_fig.suptitle('Initial Baseline Subtraction')
        baseline_ax_list = []
        sb_fig = plt.figure(figsize=(13, 13))
        sb_fig.suptitle('Smooth Baseline')
        sb_ax_list = []
        sw_fig = plt.figure(figsize=(13, 13))
        sw_fig.suptitle('Smooth Waveform')
        sw_ax_list = []
        sb_sub_fig = plt.figure(figsize=(13, 13))
        sb_sub_fig.suptitle('Smooth Baseline Subtracted Waveform')
        sb_sub_ax_list = []
        for iax in range(n_plots):
            x = np.floor(np.sqrt(n_plots))
            y = np.ceil(np.sqrt(n_plots))
            ax = baseline_fig.add_subplot(x, y, iax+1)
            baseline_ax_list.append(ax)
            ax = sb_fig.add_subplot(x, y, iax+1)
            sb_ax_list.append(ax)
            ax = sw_fig.add_subplot(x, y, iax+1)
            sw_ax_list.append(ax)
            ax = sb_sub_fig.add_subplot(x, y, iax+1)
            sb_sub_ax_list.append(ax)

        # fr_fig = plt.figure(figsize=(6, 6))
        # fr_ax = fr_fig.add_subplot(1, 1, 1)
        # fr_ax.set_title('Filter Frequency Response')
        # fr_ax.set_xlabel(r'Frequency (Hz)')

        area_fig = plt.figure(figsize=(6, 6))
        area_ax = area_fig.add_subplot(1, 1, 1)
        area_ax.set_title('Pulse Area Spectrum')
        area_ax.set_xlabel('ADC')
        area_ax.set_ylabel('N')

        height_fig = plt.figure(figsize=(6, 6))
        height_ax = height_fig.add_subplot(1, 1, 1)
        height_ax.set_title('Pulse Height Spectrum')
        height_ax.set_xlabel('ADC')
        height_ax.set_ylabel('N')

        times_fig = plt.figure(figsize=(6, 6))
        times_ax = times_fig.add_subplot(1, 1, 1)
        times_ax.set_title('T0 - PeakTime')
        times_ax.set_xlabel('Time (ns)')
        times_ax.set_ylabel('N')

        avg_wf_fig = plt.figure(figsize=(6, 6))
        avg_wf_ax = avg_wf_fig.add_subplot(1, 1, 1)
        avg_wf_ax.set_title('Mean Pulse')
        avg_wf_ax.set_xlabel('Time (ns)')
        avg_wf_ax.set_ylabel('ADC')

        global_wf_fig = plt.figure(figsize=(6, 6))
        global_wf_ax = global_wf_fig.add_subplot(1, 1, 1)
        global_wf_ax.set_title('Global Mean Pulse')
        global_wf_ax.set_xlabel('Time (ns)')
        global_wf_ax.set_ylabel('ADC')

        # Plot Waveforms
        base_handles = None
        sb_handles = None
        sw_handles = None
        sb_sub_handles = None
        for ipix in range(n_plots):
            base_ax = baseline_ax_list[ipix]
            base_ax.plot(dl0[ipix], label="raw")
            base_ax.plot(baseline_sub[ipix], label="subtracted")
            base_ax.plot([iw_l, iw_l], base_ax.get_ylim(), color='r')
            base_ax.plot([iw_r, iw_r], base_ax.get_ylim(), color='r')
            base_handles = base_ax.get_legend_handles_labels()
            # base_ax.legend(loc=1)

            sb_ax = sb_ax_list[ipix]
            sb_ax.plot(no_pulse[ipix], label="baseline")
            sb_ax.plot(smooth_baseline[ipix], label="smooth baseline")
            sb_ax.plot([iw_l, iw_l], sb_ax.get_ylim(), color='r')
            sb_ax.plot([iw_r, iw_r], sb_ax.get_ylim(), color='r')
            sb_ax.plot([pw_l, pw_l], sb_ax.get_ylim(), color='g')
            sb_ax.plot([pw_r, pw_r], sb_ax.get_ylim(), color='g')
            sb_handles = sb_ax.get_legend_handles_labels()
            # sb_ax.legend(loc=1)

            sw_ax = sw_ax_list[ipix]
            sw_ax.plot(baseline_sub[ipix], label="before")
            sw_ax.plot(smooth_wf[ipix], label="smoothed")
            sw_ax.plot([iw_l, iw_l], sw_ax.get_ylim(), color='r')
            sw_ax.plot([iw_r, iw_r], sw_ax.get_ylim(), color='r')
            sw_handles = sw_ax.get_legend_handles_labels()
            # sw_ax.legend(loc=1)

            sb_sub_ax = sb_sub_ax_list[ipix]
            sb_sub_ax.plot(dl0[ipix], label="raw", alpha=0.4)
            sb_sub_ax.plot(smooth_wf[ipix], label="before")
            sb_sub_ax.plot(sb_sub_wf[ipix], label="smooth-baseline subtracted")
            sb_sub_ax.plot([iw_l, iw_l], sb_sub_ax.get_ylim(), color='r')
            sb_sub_ax.plot([iw_r, iw_r], sb_sub_ax.get_ylim(), color='r')
            sb_sub_handles = sb_sub_ax.get_legend_handles_labels()
            # sb_sub_ax.legend(loc=1)

        # Plot legends
        baseline_fig.legend(*base_handles, loc=1)
        sb_fig.legend(*sb_handles, loc=1)
        sw_fig.legend(*sw_handles, loc=1)
        sb_sub_fig.legend(*sb_sub_handles, loc=1)

        # Plot average wf
        avg_wf_ax.plot(avg_wf, label="Average Waveform")
        avg_wf_ax.plot(fit_wf, label="Gaussian Fit")
        avg_wf_ax.plot([iw_l, iw_l], avg_wf_ax.get_ylim(), color='r', alpha=1)
        avg_wf_ax.plot([iw_r, iw_r], avg_wf_ax.get_ylim(), color='r', alpha=1)
        avg_wf_ax.plot([pw_l, pw_l], avg_wf_ax.get_ylim(), color='g', alpha=1)
        avg_wf_ax.plot([pw_r, pw_r], avg_wf_ax.get_ylim(), color='g', alpha=1)
        avg_wf_ax.legend(loc=1)

        # Save figures
        fig_dir = join(self.file_reader.output_directory,
                       "extract_pulse_spectrum")
        if not exists(fig_dir):
            self.log.info("Creating directory: {}".format(fig_dir))
            makedirs(fig_dir)

        baseline_path = join(fig_dir, "initial_baseline_subtraction.pdf")
        sb_path = join(fig_dir, "smooth_baseline.pdf")
        sw_path = join(fig_dir, "smooth_wf.pdf")
        sb_sub_path = join(fig_dir, "smooth_baseline_subtracted.pdf")
        avg_wf_path = join(fig_dir, "avg_wf.pdf")
        area_path = join(fig_dir, "area.pdf")
        height_path = join(fig_dir, "height.pdf")
        times_path = join(fig_dir, "times.pdf")
        global_path = join(fig_dir, "global.pdf")

        baseline_fig.savefig(baseline_path)
        self.log.info("Created figure: {}".format(baseline_path))
        sb_fig.savefig(sb_path)
        self.log.info("Created figure: {}".format(sb_path))
        sw_fig.savefig(sw_path)
        self.log.info("Created figure: {}".format(sw_path))
        sb_sub_fig.savefig(sb_sub_path)
        self.log.info("Created figure: {}".format(sb_sub_path))
        avg_wf_fig.savefig(avg_wf_path)
        self.log.info("Created figure: {}".format(avg_wf_path))

        # Prepare storage array
        area = np.zeros((n_events, n_pixels))
        height = np.zeros((n_events, n_pixels))
        times = np.zeros((n_events, n_pixels))
        global_ = np.zeros((n_events, n_samples))

        source = self.file_reader.read()
        desc = "Looping through file"
        with tqdm(total=n_events, desc=desc) as pbar:
            for event in source:
                pbar.update(1)
                index = event.count

                self.r1.calibrate(event)
                self.dl0.reduce(event)

                telid = list(event.r0.tels_with_data)[0]
                dl0 = np.copy(event.dl0.tel[telid].pe_samples[0])

                # Perform CHECM Waveform Cleaning
                sb_sub_wf, t0 = self.cleaner.apply(dl0)

                # Perform CHECM Charge Extraction
                peak_area, peak_height = self.extractor.extract(sb_sub_wf, t0)

                area[index] = peak_area
                height[index] = peak_height
                # times[index] = peak_t
                global_[index] = np.mean(dl0, axis=0)

        # Make spectrum histogram
        area_ax.hist(area[:, 817], bins=40, range=[-20, 60])
        height_ax.hist(height[:, 817], bins=40, range=[-5, 15])
        times_ax.hist(times.ravel(), bins=40)

        # Make global wf
        global_wf_ax.plot(np.mean(global_, axis=0))

        area_fig.savefig(area_path)
        self.log.info("Created figure: {}".format(area_path))
        height_fig.savefig(height_path)
        self.log.info("Created figure: {}".format(height_path))
        times_fig.savefig(times_path)
        self.log.info("Created figure: {}".format(times_path))
        global_wf_fig.savefig(global_path)
        self.log.info("Created figure: {}".format(global_path))

        numpy_path = join(fig_dir, "height_area.npz")
        np.savez(numpy_path, height=height, area=area)
        self.log.info("Created numpy file: {}".format(numpy_path))

    def finish(self):
        pass


if __name__ == '__main__':
    exe = EventFileLooper()
    exe.run()
