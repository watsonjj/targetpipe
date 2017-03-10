import iminuit
import numpy as np
from scipy.stats import norm as normal
from ctapipe.core import Component
from targetpipe.fitting.mapm_spe import mapm_spe_fit, pedestal_signal, \
    pe_signal
from scipy.stats.distributions import poisson


class CHECMFitterSPE(Component):
    name = 'CHECMFitterSPE'

    def __init__(self, config, tool, **kwargs):
        """
        Generic fitter for spe illumination of MAPMs

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

        self.hist = None
        self.edges = None
        self.between = None
        self.fit = None
        self.coeff = None
        self.fit_x = None

        self.k = np.arange(1, 11)
        self.subfit_labels = ['pedestal']
        for pe in self.k:
            self.subfit_labels.append('{}pe'.format(pe))

        self.coeff_list = ['norm', 'eped', 'eped_sigma',
                           'spe', 'spe_sigma', 'lambda_']

        self.nbins = 60
        # self.range = [-5, 15]
        self.range = [-30, 100]
        self.initial = dict(norm=20000,
                            eped=0,
                            eped_sigma=5,
                            spe=20,
                            spe_sigma=20,
                            lambda_=1)
        self.limits = dict(limit_norm=(0, 100000),
                           limit_eped=(-10, 10),
                           limit_eped_sigma=(0, 100),
                           limit_spe=(0, 90),
                           limit_spe_sigma=(0, 100),
                           limit_lambda_=(0, 10))

    @property
    def gain(self):
        return self._get_gain()

    @property
    def gain_error(self):
        return self._get_gain_error()

    @property
    def subfits(self):
        return self._get_subfits()

    def apply(self, spectrum):
        hist, edges = self.get_histogram(spectrum)
        between = (edges[1:] + edges[:-1]) / 2
        fit_x = np.linspace(edges[0], edges[-1], edges.size*10)

        self.hist = hist
        self.edges = edges
        self.between = between
        self.fit_x = fit_x

        try:
            fit, coeff = self._fit(hist, between, fit_x)
        except RuntimeError:
            self.fit = [0]*len(fit_x)
            self.coeff = {}
            return False

        self.fit = fit
        self.coeff = coeff
        return True

    def get_histogram(self, spectrum):
        hist, edges = np.histogram(spectrum, bins=self.nbins, range=self.range)
        return hist, edges

    def _get_gain(self):
        return self.coeff['spe']

    def _get_gain_error(self):
        return np.sqrt(self.coeff['spe_sigma'])

    def _fit(self, hist, between, fit_x):
        p0 = self.initial
        limits = self.limits
        fit, coeff = self.iminuit_fit(between, hist, p0, fit_x, limits)
        return fit, coeff

    @staticmethod
    def iminuit_fit(x, y, p0, fit_x=None, limits=None):
        if fit_x is None:
            fit_x = x
        if limits is None:
            limits = {}

        def minimizehist(norm, eped, eped_sigma, spe, spe_sigma, lambda_):
            p = mapm_spe_fit(x, norm, eped, eped_sigma, spe, spe_sigma,
                             lambda_)
            like = -2 * poisson.logpmf(y, p)
            return np.sum(like)

        m0 = iminuit.Minuit(minimizehist, **p0, **limits,
                            print_level=0, pedantic=False, throw_nan=True)
        m0.migrad()

        result = mapm_spe_fit(fit_x, **m0.values)
        return result, m0.values

    def _get_subfits(self):
        if self.coeff:
            def pedestal_kw(x, norm, eped, eped_sigma, lambda_, **kw):
                return pedestal_signal(x, norm, eped, eped_sigma, lambda_)

            def pe_kw(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_, **kw):
                return pe_signal(self.k[:, None], x[None, :], norm, eped,
                                 eped_sigma, spe, spe_sigma, lambda_)

            pedestal = pedestal_kw(self.fit_x, **self.coeff)
            pe_fits = pe_kw(self.fit_x, **self.coeff)
            subfits = dict(pedestal=pedestal)
            for i, pe_i in enumerate(self.k):
                subfits['{}pe'.format(pe_i)] = pe_fits[i]
        else:
            subfits = dict(pedestal=np.zeros(self.fit_x.shape))
            for i, pe_i in enumerate(self.k):
                subfits['{}pe'.format(pe_i)] = np.zeros(self.fit_x.shape)

        return subfits


class CHECMFitterBright(Component):
    name = 'CHECMFitterBright'

    def __init__(self, config, tool, **kwargs):
        """
        Generic fitter for bright illumination of MAPMs

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

        self._bright_mean = None
        self._bright_std = None

        self.hist = None
        self.edges = None
        self.between = None
        self.fit = None
        self.coeff = None
        self.fit_x = None

        self.k = np.arange(1, 11)
        self.subfit_labels = []

        self.coeff_list = ['norm', 'mean', 'stddev']

        self.nbins = 40
        self.range = [None, None]
        self.initial = dict(norm=None,
                            mean=None,
                            stddev=None)
        self.limits = dict(limit_norm=(0, 100000),
                           limit_mean=(-10000, 10000),
                           limit_stddev=(0, 10000))

    @property
    def gain(self):
        return self._get_gain()

    @property
    def gain_error(self):
        return self._get_gain_error()

    @property
    def subfits(self):
        return dict()

    def apply(self, spectrum):
        hist, edges = self.get_histogram(spectrum)
        between = (edges[1:] + edges[:-1]) / 2
        fit_x = np.linspace(edges[0], edges[-1], edges.size*10)

        self.hist = hist
        self.edges = edges
        self.between = between
        self.fit_x = fit_x

        try:
            fit, coeff = self._fit(hist, between, fit_x)
        except RuntimeError:
            self.fit = [0]*len(fit_x)
            self.coeff = {}
            return False

        self.fit = fit
        self.coeff = coeff
        return True

    def get_histogram(self, spectrum):
        nonoutliers = spectrum[abs(spectrum - spectrum.mean())
                               <= 2 * spectrum.std()]
        self._bright_mean = np.mean(nonoutliers)
        self._bright_std = np.std(nonoutliers)
        if not self.range[0]:
            self.range[0] = self._bright_mean - 3 * self._bright_std
        if not self.range[1]:
            self.range[1] = self._bright_mean + 3 * self._bright_std
        hist, edges = np.histogram(spectrum, bins=self.nbins, range=self.range)
        return hist, edges

    def _get_gain(self):
        return self.coeff['mean']

    def _get_gain_error(self):
        return np.sqrt(self.coeff['stddev'])

    def _fit(self, hist, between, fit_x):
        p0 = self.initial
        limits = self.limits
        if not p0['norm']:
            p0['norm'] = hist.sum()
        if not p0['mean']:
            p0['mean'] = self._bright_mean
        if not p0['stddev']:
            p0['stddev'] = self._bright_std
        fit, coeff = self.iminuit_fit(between, hist, p0, fit_x, limits)
        return fit, coeff

    @staticmethod
    def iminuit_fit(x, y, p0, fit_x=None, limits=None):
        if fit_x is None:
            fit_x = x
        if limits is None:
            limits = {}

        def gaus(x_, norm, mean, stddev):
            return norm * normal.pdf(x_, mean, stddev)

        def minimizehist(norm, mean, stddev):
            p = gaus(x, norm, mean, stddev)
            like = -2 * poisson.logpmf(y, p)
            return np.sum(like)

        m0 = iminuit.Minuit(minimizehist, **p0, **limits,
                            print_level=0, pedantic=False, throw_nan=True)
        m0.migrad()

        result = gaus(fit_x, **m0.values)
        return result, m0.values
