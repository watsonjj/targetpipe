from abc import abstractmethod
import iminuit
import numpy as np
from scipy.stats import norm as normal
from traitlets import CaselessStrEnum

from ctapipe.core import Component, Factory
from targetpipe.fitting.spe_mapm import mapm_spe_fit, pedestal_signal, \
    pe_signal
from scipy.stats.distributions import poisson


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
        self.fit = None
        self.coeff = None
        self.fit_x = None

        self.nbins = 100
        self.range = [-10, 100]

        self.coeff_list = []
        self.initial = dict()
        self.limits = dict()
        self.subfit_labels = []

    @property
    def subfits(self):
        return dict()

    def add_parameter(self, name, inital, lower, upper):
        self.coeff_list.append(name)
        self.initial[name] = inital
        self.limits["limit_" + name] = (lower, upper)

    def apply(self, spectrum):
        hist, edges = self.get_histogram(spectrum)
        between = (edges[1:] + edges[:-1]) / 2
        fit_x = np.linspace(edges[0], edges[-1], edges.size*10)

        self.hist = hist
        self.edges = edges
        self.between = between
        self.fit_x = fit_x

        try:
            fit, coeff = self._fit(hist, edges, between, fit_x)
        except RuntimeError:
            self.fit = [0]*len(fit_x)
            self.coeff = {}
            return False

        self.fit = fit
        self.coeff = coeff
        return True

    def get_histogram(self, spectrum):
        range_ = self.range[:]
        hist, edges = np.histogram(spectrum, bins=self.nbins, range=range_)
        return hist, edges

    @abstractmethod
    def _fit(self, hist, edges, between, fit_x):
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
        fit_x : ndarray
            X coordinates for the fit curve.

        Returns
        -------
        fit : ndarray
            Y coordinates for the fit curve.
        coeff : dict
            Values for the parameters of the fit.
        """

    @staticmethod
    @abstractmethod
    def iminuit_fit(x, y, p0, fit_x=None, limits=None):
        """
        Implementation of the fit using the `iminuit` minimisation package.
        
        Parameters
        ----------
        x : ndarray
            X values of the distribution.
        y : ndarray
            Y values of the distribution.
        p0 : dict
            Initial values for the fit parameters.
        fit_x : ndarray
            X coordinates for the fit curve.
        limits : dict
            Limits for the fit parameters.

        Returns
        -------
        result : ndarray
            Y coordinates for the fit curve.
        m0.values : dict
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
        self.add_parameter("lambda_", 20, 0, 10)

        self.k = np.arange(1, 11)
        self.subfit_labels = ['pedestal']
        for pe in self.k:
            self.subfit_labels.append('{}pe'.format(pe))

    @property
    def subfits(self):
        return self._get_subfits()

    def _fit(self, hist, edges, between, fit_x):
        p0 = self.initial.copy()
        limits = self.limits.copy()
        if p0['norm'] is None:
            p0['norm'] = np.sum(np.diff(edges) * hist)
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


class CHECBrightFitter(ChargeFitter):
    name = 'CHECBrightFitter'
    fitter_type = 'bright'

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
        super().__init__(config=config, tool=tool, **kwargs)

        self.nbins = 40
        self.range = [None, None]

        self.add_parameter('norm', None, 0, 100000)
        self.add_parameter('mean', None, None, None)
        self.add_parameter('stddev', None, 0, None)

        self._bright_mean = None
        self._bright_std = None

    def get_histogram(self, spectrum):
        nonoutliers = spectrum[abs(spectrum - spectrum.mean())
                               <= 2 * spectrum.std()]
        self._bright_mean = np.mean(nonoutliers)
        self._bright_std = np.std(nonoutliers)
        range_ = self.range[:]
        if not range_[0]:
            range_[0] = self._bright_mean - 2.5 * self._bright_std
        if not range_[1]:
            range_[1] = self._bright_mean + 2.5 * self._bright_std
        hist, edges = np.histogram(spectrum, bins=self.nbins, range=range_)
        return hist, edges

    def _fit(self, hist, edges, between, fit_x):
        p0 = self.initial.copy()
        limits = self.limits.copy()
        if not p0['norm']:
            p0['norm'] = np.sum(np.diff(edges) * hist)
        if not p0['mean']:
            p0['mean'] = self._bright_mean
        if not p0['stddev']:
            p0['stddev'] = self._bright_std/2
        if None in limits['limit_mean']:
            limits['limit_mean'] = (edges[0], edges[-1])
        if None in limits['limit_stddev']:
            limits['limit_stddev'] = (0, np.abs(edges[-1]))
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
        self.add_parameter("eped", 0, -10, 10)
        self.add_parameter("eped_sigma", 5, 0, 100)
        self.add_parameter("spe", 20, 0, 90)
        self.add_parameter("spe_sigma", 20, 0, 100)
        self.add_parameter("lambda_", 20, 0, 10)

        self.k = np.arange(1, 11)
        self.subfit_labels = ['pedestal']
        for pe in self.k:
            self.subfit_labels.append('{}pe'.format(pe))

    @property
    def subfits(self):
        return self._get_subfits()

    def _fit(self, hist, edges, between, fit_x):
        p0 = self.initial.copy()
        limits = self.limits.copy()
        if p0['norm'] is None:
            p0['norm'] = np.sum(np.diff(edges) * hist)
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
