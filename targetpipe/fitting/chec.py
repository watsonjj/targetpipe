from abc import abstractmethod
import iminuit
import numpy as np
from scipy.stats import norm as normal
from traitlets import CaselessStrEnum

from ctapipe.core import Component, Factory
from targetpipe.fitting.spe_mapm import mapm_spe_fit, \
    pedestal_signal as mapm_pedestal, pe_signal as mapm_pe
from targetpipe.fitting.spe_sipm import sipm_spe_fit, \
    pedestal_signal as sipm_pedestal, pe_signal as sipm_pe
from scipy.stats.distributions import poisson
from scipy.stats import chisquare


class ChargeFitter(Component):
    name = 'ChargeFitter'
    fitter_type = 'base'

    def __init__(self, config, tool, **kwargs):
        """
        Base class for fitters of charge distributions

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
        self.coeff = None

        self.nbins = 100
        self.range = [-10, 100]

        self.coeff_list = []
        self.initial = dict()
        self.limits = dict()
        self.subfit_labels = []

    @property
    def fit_x(self):
        return np.linspace(self.edges[0], self.edges[-1], 10*self.edges.size)

    @property
    def fit(self):
        y = self.fit_function(self.fit_x, **self.coeff)
        return y

    @property
    def chi2(self):
        h = np.hstack(self.hist)
        f = np.hstack(self.fit_function(self.between, **self.coeff))
        b = h >= 5
        h = h[b]
        f = f[b]
        chi2 = np.sum(np.power(h - f, 2)/f)
        return chi2

    @property
    def dof(self):
        h = np.hstack(self.hist)
        n = h[h >= 5].size
        m = len(self.coeff)
        dof = n - 1 - m
        return dof

    @property
    def reduced_chi2(self):
        return self.chi2 / self.dof

    @property
    def p_value(self):
        h = np.hstack(self.hist)
        f = np.hstack(self.fit_function(self.between, **self.coeff))
        b = h >= 5
        h = h[b]
        f = f[b]
        return chisquare(h, f, len(self.coeff)).pvalue

    def add_parameter(self, name, inital, lower, upper):
        self.coeff_list.append(name)
        self.initial[name] = inital
        self.limits["limit_" + name] = (lower, upper)

    def apply(self, spectrum):
        hist, edges = self.get_histogram(spectrum)
        between = (edges[1:] + edges[:-1]) / 2

        self.hist = hist
        self.edges = edges
        self.between = between

        try:
            coeff = self._perform_fit(hist, edges, between)
        except RuntimeError:
            self.coeff = {}
            return False

        self.coeff = coeff
        return True

    def get_histogram(self, spectrum):
        range_ = self.range[:]
        hist, edges = np.histogram(spectrum, bins=self.nbins, range=range_)
        return hist, edges

    @abstractmethod
    def _perform_fit(self, hist, edges, between):
        """
        Function that performs the fit on the histogram of charges

        Parameters
        ----------
        hist : ndarray
            Histogram of charges. Result of `get_histogram`
        edges : ndarray
            Edges of the bins for hist. Result of `get_histogram`
        between : ndarray
            Centers of the bins for hist.

        Returns
        -------
        coeff : dict
            Values for the parameters of the fit.
        """

class CHECMSPEFitter(ChargeFitter):
    name = 'CHECMSPEFitter'
    fitter_type = 'spe'

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
        super().__init__(config=config, tool=tool, **kwargs)

        self.nbins = 60
        self.range = [-30, 100]

        self.add_parameter("norm", None, 0, 100000)
        self.add_parameter("eped", 0, -10, 10)
        self.add_parameter("eped_sigma", 5, 0, 100)
        self.add_parameter("spe", 20, 0, 90)
        self.add_parameter("spe_sigma", 20, 0, 100)
        self.add_parameter("lambda_", 1, 0, 10)

        self.pedestal_signal = mapm_pedestal
        self.pe_signal = mapm_pe

        self.k = np.arange(1, 11)
        self.subfit_labels = ['pedestal']
        for pe in self.k:
            self.subfit_labels.append('{}pe'.format(pe))

    # @property
    # def subfits(self):
    #     return self._get_subfits()

    @staticmethod
    def fit_function(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_):
        return mapm_spe_fit(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_)

    def _perform_fit(self, hist, edges, between):
        p0 = self.initial.copy()
        limits = self.limits.copy()
        if p0['norm'] is None:
            p0['norm'] = np.sum(np.diff(edges) * hist)
        coeff = self.iminuit_fit(between, hist, p0, limits)
        return coeff

    def iminuit_fit(self, x, y, p0, limits=None):
        if limits is None:
            limits = {}

        def minimizehist(norm, eped, eped_sigma, spe, spe_sigma, lambda_):
            p = self.fit_function(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_)
            like = -2 * poisson.logpmf(y, p)
            return np.sum(like)

        m0 = iminuit.Minuit(minimizehist, **p0, **limits,
                            print_level=0, pedantic=False, throw_nan=True)
        m0.migrad()
        return m0.values

    # def _get_subfits(self):
    #     if self.coeff:
    #         def pedestal_kw(x, norm, eped, eped_sigma, lambda_, **kw):
    #             return self.pedestal_signal(x, norm, eped, eped_sigma, lambda_)
    #
    #         def pe_kw(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_, **kw):
    #             return self.pe_signal(self.k[:, None], x[None, :], norm, eped,
    #                                   eped_sigma, spe, spe_sigma, lambda_)
    #
    #         pedestal = pedestal_kw(self.fit_x, **self.coeff)
    #         pe_fits = pe_kw(self.fit_x, **self.coeff)
    #         subfits = dict(pedestal=pedestal)
    #         for i, pe_i in enumerate(self.k):
    #             subfits['{}pe'.format(pe_i)] = pe_fits[i]
    #     else:
    #         subfits = dict(pedestal=np.zeros(self.fit_x.shape))
    #         for i, pe_i in enumerate(self.k):
    #             subfits['{}pe'.format(pe_i)] = np.zeros(self.fit_x.shape)
    #
    #     return subfits


# class CHECBrightFitter(ChargeFitter):
#     name = 'CHECBrightFitter'
#     fitter_type = 'bright'
#
#     def __init__(self, config, tool, **kwargs):
#         """
#         Generic fitter for bright illumination of MAPMs
#
#         Parameters
#         ----------
#         config : traitlets.loader.Config
#             Configuration specified by config file or cmdline arguments.
#             Used to set traitlet values.
#             Set to None if no configuration to pass.
#         tool : ctapipe.core.Tool
#             Tool executable that is calling this component.
#             Passes the correct logger to the component.
#             Set to None if no Tool to pass.
#         kwargs
#         """
#         super().__init__(config=config, tool=tool, **kwargs)
#
#         self.nbins = 40
#         self.range = [None, None]
#
#         self.add_parameter('norm', None, 0, 100000)
#         self.add_parameter('mean', None, None, None)
#         self.add_parameter('stddev', None, 0, None)
#
#         self._bright_mean = None
#         self._bright_std = None
#
#     def get_histogram(self, spectrum):
#         nonoutliers = spectrum[abs(spectrum - spectrum.mean())
#                                <= 2 * spectrum.std()]
#         self._bright_mean = np.mean(nonoutliers)
#         self._bright_std = np.std(nonoutliers)
#         range_ = self.range[:]
#         if not range_[0]:
#             range_[0] = self._bright_mean - 2.5 * self._bright_std
#         if not range_[1]:
#             range_[1] = self._bright_mean + 2.5 * self._bright_std
#         hist, edges = np.histogram(spectrum, bins=self.nbins, range=range_)
#         return hist, edges
#
#     def _perform_fit(self, hist, edges, between, fit_x):
#         p0 = self.initial.copy()
#         limits = self.limits.copy()
#         if not p0['norm']:
#             p0['norm'] = np.sum(np.diff(edges) * hist)
#         if not p0['mean']:
#             p0['mean'] = self._bright_mean
#         if not p0['stddev']:
#             p0['stddev'] = self._bright_std/2
#         if None in limits['limit_mean']:
#             limits['limit_mean'] = (edges[0], edges[-1])
#         if None in limits['limit_stddev']:
#             limits['limit_stddev'] = (0, np.abs(edges[-1]))
#         fit, coeff = self.iminuit_fit(between, hist, p0, fit_x, limits)
#         return fit, coeff
#
#     @staticmethod
#     def iminuit_fit(x, y, p0, fit_x=None, limits=None):
#         if fit_x is None:
#             fit_x = x
#         if limits is None:
#             limits = {}
#
#         def gaus(x_, norm, mean, stddev):
#             return norm * normal.pdf(x_, mean, stddev)
#
#         def minimizehist(norm, mean, stddev):
#             p = gaus(x, norm, mean, stddev)
#             like = -2 * poisson.logpmf(y, p)
#             return np.sum(like)
#
#         m0 = iminuit.Minuit(minimizehist, **p0, **limits,
#                             print_level=0, pedantic=False, throw_nan=True)
#         m0.migrad()
#
#         result = gaus(fit_x, **m0.values)
#         return result, m0.values


class CHECSSPEFitter(ChargeFitter):
    name = 'CHECSSPEFitter'
    fitter_type = 'spe'

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
        super().__init__(config=config, tool=tool, **kwargs)

        self.nbins = 60
        self.range = [-30, 100]

        self.add_parameter("norm", None, 0, 100000)
        self.add_parameter("eped", -0.5, -1, 1)
        self.add_parameter("eped_sigma", 0.5, 0, 5)
        self.add_parameter("spe", 1, 0, 2)
        self.add_parameter("spe_sigma", 0.1, 0, 1)
        self.add_parameter("lambda_", 1, 0, 5)
        self.add_parameter("opct", 0.5, 0, 1)
        self.add_parameter("pap", 0.5, 0, 1)
        self.add_parameter("dap", 0.5, 0, 1)

        self.pedestal_signal = sipm_pedestal
        self.pe_signal = sipm_pe

        self.k = np.arange(1, 11)
        self.subfit_labels = ['pedestal']
        for pe in self.k:
            self.subfit_labels.append('{}pe'.format(pe))

    @staticmethod
    def fit_function(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_, opct, pap, dap):
        return sipm_spe_fit(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_, opct, pap, dap)

    def _perform_fit(self, hist, edges, between):
        p0 = self.initial.copy()
        limits = self.limits.copy()
        if p0['norm'] is None:
            p0['norm'] = np.sum(np.diff(edges) * hist)
        coeff = self.iminuit_fit(between, hist, p0, limits)
        return coeff

    def iminuit_fit(self, x, y, p0, limits=None):
        if limits is None:
            limits = {}

        def minimizehist(norm, eped, eped_sigma, spe, spe_sigma, lambda_, opct, pap, dap):
            p = self.fit_function(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_, opct, pap, dap)
            like = -2 * poisson.logpmf(y, p)
            return np.nansum(like)

        m0 = iminuit.Minuit(minimizehist, **p0, **limits,
                            print_level=0, pedantic=False, throw_nan=False)
        m0.migrad()

        return m0.values

    # def _get_subfits(self):
    #     if self.coeff:
    #         def pedestal_kw(x, norm, eped, eped_sigma, lambda_, **kw):
    #             return self.pedestal_signal(x, norm, eped, eped_sigma, lambda_)
    #
    #         def pe_kw(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_, opct, pap, dap, **kw):
    #             return self.pe_signal(self.k[:, None], x[None, :], norm, eped,
    #                                   eped_sigma, spe, spe_sigma, lambda_, opct, pap, dap)
    #
    #         pedestal = pedestal_kw(self.fit_x, **self.coeff)
    #         pe_fits = pe_kw(self.fit_x, **self.coeff)
    #         subfits = dict(pedestal=pedestal)
    #         for i, pe_i in enumerate(self.k):
    #             subfits['{}pe'.format(pe_i)] = pe_fits[i]
    #     else:
    #         subfits = dict(pedestal=np.zeros(self.fit_x.shape))
    #         for i, pe_i in enumerate(self.k):
    #             subfits['{}pe'.format(pe_i)] = np.zeros(self.fit_x.shape)
    #
    #     return subfits


class CHECSSPEMultiFitter(ChargeFitter):
    name = 'CHECSSPEMultiFitter'
    fitter_type = 'spe'

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

        self.nbins = 60
        self.range = [-30, 100]

        self.add_parameter("norm1", None, 0, 100000)
        self.add_parameter("norm2", None, 0, 100000)
        self.add_parameter("norm3", None, 0, 100000)
        self.add_parameter("eped", -0.5, -1, 1)
        self.add_parameter("eped_sigma", 0.5, 0, 5)
        self.add_parameter("spe", 1, 0, 2)
        self.add_parameter("spe_sigma", 0.1, 0, 1)
        self.add_parameter("lambda_1", 1, 0.5, 5)
        self.add_parameter("lambda_2", 1, 0.5, 5)
        self.add_parameter("lambda_3", 1, 0.5, 5)
        self.add_parameter("opct", 0.5, 0, 1)
        self.add_parameter("pap", 0.5, 0, 1)
        self.add_parameter("dap", 0.5, 0, 1)

        self.pedestal_signal = sipm_pedestal
        self.pe_signal = sipm_pe

        self.k = np.arange(1, 11)
        self.subfit_labels = ['pedestal']
        for pe in self.k:
            self.subfit_labels.append('{}pe'.format(pe))

    def apply_multi(self, spectrum1, spectrum2, spectrum3):
        hist1, edges = self.get_histogram(spectrum1)
        hist2, _ = self.get_histogram(spectrum2)
        hist3, _ = self.get_histogram(spectrum3)

        between = (edges[1:] + edges[:-1]) / 2

        hist = [hist1, hist2, hist3]
        self.hist = hist
        self.edges = edges
        self.between = between

        try:
            coeff = self._perform_fit(hist, edges, between)
        except RuntimeError:
            self.coeff = {}
            return False

        self.coeff = coeff
        return True

    def apply(self, spectrum):
        self.log.warn("No apply method for this fitter, use apply_multi")

    @staticmethod
    def fit_function(x, norm1, norm2, norm3, eped, eped_sigma, spe, spe_sigma, lambda_1, lambda_2, lambda_3, opct, pap, dap):
        fit = sipm_spe_fit
        p1 = fit(x, norm1, eped, eped_sigma, spe, spe_sigma, lambda_1, opct, pap, dap)
        p2 = fit(x, norm2, eped, eped_sigma, spe, spe_sigma, lambda_2, opct, pap, dap)
        p3 = fit(x, norm3, eped, eped_sigma, spe, spe_sigma, lambda_3, opct, pap, dap)
        return p1, p2, p3

    def _perform_fit(self, hist, edges, between):
        p0 = self.initial.copy()
        limits = self.limits.copy()
        if p0['norm1'] is None:
            p0['norm1'] = np.sum(np.diff(edges) * hist[0])
        if p0['norm2'] is None:
            p0['norm2'] = np.sum(np.diff(edges) * hist[1])
        if p0['norm3'] is None:
            p0['norm3'] = np.sum(np.diff(edges) * hist[2])
        coeff = self.iminuit_fit(between, hist, p0, limits)
        return coeff

    def iminuit_fit(self, x, y, p0, limits=None):
        fit = sipm_spe_fit
        if limits is None:
            limits = {}

        def minimizehist(norm1, norm2, norm3, eped, eped_sigma, spe, spe_sigma, lambda_1, lambda_2, lambda_3, opct, pap, dap):
            p1, p2, p3 = self.fit_function(x, norm1, norm2, norm3, eped, eped_sigma, spe, spe_sigma, lambda_1, lambda_2, lambda_3, opct, pap, dap)
            like1 = -2 * poisson.logpmf(y[0], p1)
            like2 = -2 * poisson.logpmf(y[1], p2)
            like3 = -2 * poisson.logpmf(y[2], p3)
            like = np.hstack([like1, like2, like3])
            return np.nansum(like)

        m0 = iminuit.Minuit(minimizehist, **p0, **limits,
                            print_level=0, pedantic=False, throw_nan=False)
        m0.migrad()

        return m0.values


class ChargeFitterFactory(Factory):
    """
    Factory to obtain a ChargeFitter.
    """
    name = "ChargeFitterFactory"
    description = "Obtain ChargeFitter based on fitter traitlet"

    subclasses = Factory.child_subclasses(ChargeFitter)
    subclass_names = [c.__name__ for c in subclasses]

    fitter = CaselessStrEnum(subclass_names, 'CHECMSPEFitter',
                             help='Charge fitter to use.').tag(config=True)

    def get_factory_name(self):
        return self.name

    def get_product_name(self):
        return self.fitter


class SPEFitterFactory(Factory):
    """
    Factory to obtain a ChargeFitter of type 'spe'.
    """
    name = "SPEFitterFactory"
    description = "Obtain ChargeFitter based on fitter traitlet"

    subclasses = Factory.child_subclasses(ChargeFitter)
    subclass_names = [c.__name__ for c in subclasses if c.fitter_type == 'spe']

    fitter = CaselessStrEnum(subclass_names, 'CHECMSPEFitter',
                             help='Charge fitter to use.').tag(config=True)

    def get_factory_name(self):
        return self.name

    def get_product_name(self):
        return self.fitter
