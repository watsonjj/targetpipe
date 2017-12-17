from abc import abstractmethod
import iminuit
import numpy as np
from fit_algorithms import sipm_spe_fit, \
    pedestal_signal as sipm_pedestal, pe_signal as sipm_pe
from scipy.stats.distributions import poisson
from scipy.stats import chisquare


class ChargeFitter:

    def __init__(self):
        self.hist = None
        self.edges = None
        self.between = None
        self.coeff = None

        self.nbins = 100
        self.range = [-10, 100]

        self.coeff_list = []
        self.initial = dict()
        self.fix = dict()
        self.p0 = None
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


class CHECSSPEFitter(ChargeFitter):

    def __init__(self):
        super().__init__()

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
        self.p0 = self.initial.copy()
        limits = self.limits.copy()
        fix = self.fix.copy()
        if self.p0['norm'] is None:
            self.p0['norm'] = np.sum(np.diff(edges) * hist)
        coeff = self.iminuit_fit(between, hist, self.p0, limits, fix)
        return coeff

    def iminuit_fit(self, x, y, p0, limits=None, fix=None):
        if limits is None:
            limits = {}
        if fix is None:
            fix = {}

        def minimizehist(norm, eped, eped_sigma, spe, spe_sigma, lambda_, opct, pap, dap):
            p = self.fit_function(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_, opct, pap, dap)
            like = -2 * poisson.logpmf(y, p)
            return np.nansum(like)

        m0 = iminuit.Minuit(minimizehist, **p0, **limits, **fix,
                            print_level=0, pedantic=False, throw_nan=False)
        m0.migrad()

        return m0.values


class CHECSSPEMultiFitter(ChargeFitter):

    def __init__(self):
        super().__init__()

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
        print("No apply method for this fitter, use apply_multi")

    @staticmethod
    def fit_function(x, norm1, norm2, norm3, eped, eped_sigma, spe, spe_sigma, lambda_1, lambda_2, lambda_3, opct, pap, dap):
        fit = sipm_spe_fit
        p1 = fit(x, norm1, eped, eped_sigma, spe, spe_sigma, lambda_1, opct, pap, dap)
        p2 = fit(x, norm2, eped, eped_sigma, spe, spe_sigma, lambda_2, opct, pap, dap)
        p3 = fit(x, norm3, eped, eped_sigma, spe, spe_sigma, lambda_3, opct, pap, dap)
        return p1, p2, p3

    def _perform_fit(self, hist, edges, between):
        self.p0 = self.initial.copy()
        limits = self.limits.copy()
        fix = self.fix.copy()
        if self.p0['norm1'] is None:
            self.p0['norm1'] = np.trapz(hist[0], between) #np.sum(np.diff(edges) * hist[0])
        if self.p0['norm2'] is None:
            self.p0['norm2'] = np.trapz(hist[1], between) #np.sum(np.diff(edges) * hist[1])
        if self.p0['norm3'] is None:
            self.p0['norm3'] = np.trapz(hist[2], between) #np.sum(np.diff(edges) * hist[2])
        coeff = self.iminuit_fit(between, hist, self.p0, limits, fix)
        return coeff

    def iminuit_fit(self, x, y, p0, limits=None, fix=None):
        fit = sipm_spe_fit
        if limits is None:
            limits = {}
        if fix is None:
            fix = {}

        def minimizehist(norm1, norm2, norm3, eped, eped_sigma, spe, spe_sigma, lambda_1, lambda_2, lambda_3, opct, pap, dap):
            p1, p2, p3 = self.fit_function(x, norm1, norm2, norm3, eped, eped_sigma, spe, spe_sigma, lambda_1, lambda_2, lambda_3, opct, pap, dap)
            like1 = -2 * poisson.logpmf(y[0], p1)
            like2 = -2 * poisson.logpmf(y[1], p2)
            like3 = -2 * poisson.logpmf(y[2], p3)
            like = np.hstack([like1, like2, like3])
            return np.nansum(like)

        m0 = iminuit.Minuit(minimizehist, **p0, **limits, **fix,
                            print_level=0, pedantic=False, throw_nan=False)
        m0.migrad()

        return m0.values
