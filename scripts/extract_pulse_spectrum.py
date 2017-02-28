from scipy.optimize import curve_fit
from traitlets import Dict, List, CaselessStrEnum as CaStEn
from ctapipe.core import Tool, Component
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.charge_extractors import ChargeExtractorFactory
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import general_gaussian
from IPython import embed
from os import makedirs
from os.path import join, exists


class BaselineSubtractor(Component):
    name = 'BaselineSubtractor'

    possible = ['simple', 'whole', 'exclude_pulse']

    subtractor = CaStEn(possible, 'simple',
                        help='Subtraction method to use').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        Use a method for baseline subtraction

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

        if self.subtractor == 'simple':
            self.get_baseline = self._simple
        elif self.subtractor == 'whole':
            self.get_baseline = self._whole
        elif self.subtractor == 'exclude_pulse':
            self.get_baseline = self._exclude_pulse

    def apply(self, samples):
        return samples - self.get_baseline(samples)

    @staticmethod
    def _simple(samples):
        return np.mean(samples[:, 0:32], axis=1)[:, None]

    @staticmethod
    def _whole(samples):
        return np.mean(samples, axis=1)[:, None]

    @staticmethod
    def _exclude_pulse(samples):
        dev = np.std(samples, axis=1)[:, None]
        samples_ma = np.ma.masked_where(samples > dev, samples)
        return np.mean(samples_ma, axis=1)[:, None]


class Fitter(Component):
    name = 'Fitter'

    def __init__(self, config, tool, **kwargs):
        """
        Use a method for baseline subtraction

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

        # self.x = None
        self.avg = None
        self.fit = None
        self.t0 = None

    def fit_average_wf(self, samples):
        self.avg = np.mean(samples, axis=0)

        self.fit = self.avg
        self.t0 = np.argmax(self.avg)

        # self.x = np.arange(self.avg.size)
        # p0 = [1., 40., 1.]  # Seeds
        # bounds = ([0., 10., 0.], [100., samples.shape[1]-10, 20.])
        # coeff, var_matrix = curve_fit(self.gauss, self.x, self.avg,
        #                               p0=p0, bounds=bounds)
        # self.fit = self.gauss(self.x, *coeff)
        # self.t0 = np.round(coeff[1]).astype(np.int)

    @staticmethod
    def gauss(x, *p):
        a, mu, sigma = p
        return a*np.exp(-(x-mu)**2/(2.*sigma**2))


class Convolver(Component):
    name = 'Convolver'

    possible = ['pulse', 'baseline']

    focus = CaStEn(possible, 'pulse', help='Convolving focus').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        Use a method to smooth waveform

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

        if self.focus == 'pulse':
            n_points = 20
            pulsew = 1
            self.kernel = general_gaussian(n_points, p=1.0, sig=pulsew)
        elif self.focus == 'baseline':
            n_points = 10
            w = 32
            self.kernel = general_gaussian(n_points, p=1.0, sig=w)

    def apply(self, samples):
        smooth0 = np.convolve(samples.ravel(), self.kernel, "same")
        smooth = np.reshape(smooth0, samples.shape)

        smooth *= (np.mean(samples, axis=1)/np.mean(smooth, axis=1))[:, None]
        return smooth


class MovingAverage(Component):
    name = 'MovingAverage'

    possible = ['pulse', 'baseline']

    focus = CaStEn(possible, 'pulse',
                   help='MovingAverage focus').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        Use a method to smooth waveform

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

        if self.focus == 'pulse':
            self.apply = self._pulse
        elif self.focus == 'baseline':
            self.apply = self._baseline

    def _pulse(self, samples):
        avg = self.moving_average(samples, 2)
        return avg

    def _baseline(self, baseline):
        avg = self.moving_average(baseline, 32)
        return avg

    @staticmethod
    def moving_average(a, n=3):
        embed()
        ret = np.cumsum(a, dtype=float, axis=1)
        ret[:, n:] = ret[:, n:] - ret[:, :-n]
        return ret[:, n - 1:] / n


class Filter(Component):
    name = 'Filter'

    def __init__(self, config, tool, **kwargs):
        """
        Use a method for filtering the signal

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

        self.fs = 1/1E-9
        self.highcut = None
        self.lowcut = 200E6
        self.order = 3

    def apply(self, samples):
        b, a = self._filter()
        filtered = signal.lfilter(b, a, samples, axis=1)
        return filtered

    def _filter(self):
        return self._butter_bandpass()

    def get_frequency_response(self):
        b, a = self._filter()
        w, h = signal.freqz(b, a, worN=2000)
        x = (self.fs * 0.5 / np.pi) * w
        y = abs(h)
        return x, y

    def _butter_bandpass(self):
        nyq = 0.5 * self.fs

        # Define type of filter
        wn, btype = None
        if self.lowcut and self.highcut:
            btype = 'band'
            wn = [self.lowcut / nyq, self.highcut / nyq]
        if self.lowcut and not self.highcut:
            btype = 'low'
            wn = self.lowcut / nyq
        elif not self.lowcut and self.highcut:
            btype = 'high'
            wn = self.highcut / nyq

        # noinspection PyTupleAssignmentBalance
        b, a = signal.butter(self.order, wn, btype=btype)
        return b, a


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
        # self.extractor = None
        self.dl1 = None

        self.init_baseline = None
        self.fitter = None
        self.baseline_smoother = None
        self.wf_smoother = None

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

        self.init_baseline = BaselineSubtractor(**kwargs, subtractor='simple')
        self.fitter = Fitter(**kwargs)
        self.baseline_smoother = Convolver(**kwargs, focus='baseline')
        self.wf_smoother = Convolver(**kwargs)
        # self.baseline_smoother = MovingAverage(**kwargs, focus='baseline')
        # self.wf_smoother = MovingAverage(**kwargs)

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

        # Subtract initial baseline
        baseline_sub = self.init_baseline.apply(dl0)

        # Fit average waveform
        self.fitter.fit_average_wf(baseline_sub)
        t0 = self.fitter.t0
        avg_wf = self.fitter.avg
        fit_wf = self.fitter.fit

        # Set Windows
        pw_l = t0-6
        pw_r = t0+6
        iw_l = t0-3
        iw_r = t0+3
        pulse_window = np.s_[pw_l:pw_r]

        # Get smooth baseline (no pulse)
        no_pulse = np.ma.array(baseline_sub, mask=False, fill_value=0)
        no_pulse.mask[:, pulse_window] = True
        smooth_baseline = self.baseline_smoother.apply(np.ma.filled(no_pulse))

        # Get smooth waveform
        smooth_wf = baseline_sub#self.wf_smoother.apply(baseline_sub)

        # Subtract smooth baseline
        sb_sub_wf = smooth_wf - smooth_baseline

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
        with tqdm(desc=desc) as pbar:
            for event in source:
                pbar.update(1)
                index = event.count

                self.r1.calibrate(event)
                self.dl0.reduce(event)

                telid = list(event.r0.tels_with_data)[0]
                dl0 = np.copy(event.dl0.tel[telid].pe_samples[0])

                # Subtract initial baseline
                baseline_sub = self.init_baseline.apply(dl0)

                # Fit average waveform
                self.fitter.fit_average_wf(baseline_sub)
                t0 = self.fitter.t0

                # Set Windows
                pw_l = t0 - 6
                pw_r = t0 + 6
                iw_l = t0 - 3
                iw_r = t0 + 5
                pulse_window = np.s_[pw_l:pw_r]
                int_window = np.s_[iw_l:iw_r]

                # Get smooth baseline (no pulse)
                no_pulse = np.ma.array(baseline_sub, mask=False)
                no_pulse.mask[pulse_window] = True
                smooth_baseline = self.baseline_smoother.apply(no_pulse)

                # # Get smooth waveform
                smooth_wf = baseline_sub#self.wf_smoother.apply(baseline_sub)

                # Subtract smooth baseline
                sb_sub_wf = smooth_wf - smooth_baseline

                # Mask pixels with large pre-pulse undershoot
                # dev = np.std(sb_sub_wf, axis=1)[:, None]
                # crosstalk_pix = np.where(sb_sub_wf[:, int_window] < -2*dev)[0]
                # sb_sub_wf = np.ma.array(sb_sub_wf, mask=False)
                # sb_sub_wf[crosstalk_pix] = -1

                # Extract charge
                peak_area = np.sum(sb_sub_wf[:, int_window], axis=1)
                peak_height = sb_sub_wf[np.arange(n_pixels), t0]
                # peak_t = np.argmax(sb_sub_wf[:, int_window], axis=1)

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
