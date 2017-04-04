from scipy.optimize import curve_fit
from traitlets import CaselessStrEnum as CaStEn, Int
from ctapipe.core import Component
import numpy as np
from scipy import signal
from scipy.signal import general_gaussian
from scipy.ndimage.filters import convolve1d


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
        return np.mean(samples[:, :32], axis=1)[:, None]

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
        smooth *= (np.std(samples, axis=1)/np.std(smooth, axis=1))[:, None]
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


class CHECMWaveformCleaner(Component):
    name = 'CHECMWaveformCleaner'

    width = Int(15, help='Define the width of the peak '
                         'window').tag(config=True)
    shift = Int(6, help='Define the shift of the peak window from the peakpos '
                        '(peakpos - shift).').tag(config=True)
    t0 = Int(None, allow_none=True,
             help='Override the value of t0').tag(config=True)

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

        self.init_baseline = BaselineSubtractor(config=config, tool=tool,
                                                subtractor='simple')
        self.fitter = Fitter(config=config, tool=tool)
        self.baseline_smoother = Convolver(config=config, tool=tool,
                                           focus='baseline')
        self.wf_smoother = Convolver(config=config, tool=tool)

        # Cleaning steps for plotting
        self.stages = {}

        self.pw_l = None
        self.pw_r = None

        if self.t0:
            self.log.info("User has set t0, extracted t0 will be overridden")

    def apply(self, samples):
        # Subtract initial baseline
        baseline_sub = self.init_baseline.apply(samples)

        # Fit average waveform
        self.fitter.fit_average_wf(baseline_sub)
        t0 = self.fitter.t0
        if self.t0:
            t0 = self.t0
        avg_wf = self.fitter.avg
        fit_wf = self.fitter.fit

        # Set Windows
        self.pw_l = t0 - self.shift
        self.pw_r = self.pw_l + self.width
        pulse_window = np.s_[self.pw_l:self.pw_r]

        # Get smooth baseline (no pulse)
        no_pulse = np.ma.array(baseline_sub, mask=False, fill_value=0)
        no_pulse.mask[:, pulse_window] = True
        smooth_baseline = self.baseline_smoother.apply(np.ma.filled(no_pulse))

        # Get smooth waveform
        smooth_wf = baseline_sub  # self.wf_smoother.apply(baseline_sub)

        # FINAL: Subtract smooth baseline
        cleaned = smooth_wf - smooth_baseline

        self.stages['0: raw'] = samples
        self.stages['1: baseline_sub'] = baseline_sub
        self.stages['2: avg_wf'] = avg_wf
        self.stages['3: fit_wf'] = fit_wf
        self.stages['4: no_pulse'] = no_pulse
        self.stages['5: smooth_baseline'] = smooth_baseline
        self.stages['6: smooth_wf'] = smooth_wf
        self.stages['7: cleaned'] = cleaned

        return cleaned, t0
