import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm as normal
from ctapipe.core import Component
from targetpipe.fitting.mapm_spe import mapm_spe_fit, pedestal_signal, pe_signal


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

        self.nbins = 40
        # self.range = [-5, 15]
        self.range = [-20, 60]
        self.initial = [200000, 0, 5, 20, 5, 1]
        self.bounds = ([0, -np.inf, 0, 0, 0, 0], [np.inf]*6)

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
            fit, coeff = self._fit(hist, edges, between, fit_x)
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

    def _fit(self, hist, edges, between, fit_x):
        p0 = self.initial
        bounds = self.bounds
        fit, coeff = self.scipy_fit(between, hist, p0, fit_x, bounds)
        coeff_dict = dict()
        for i, c in enumerate(self.coeff_list):
            coeff_dict[c] = coeff[i]
        return fit, coeff_dict

    @staticmethod
    def scipy_fit(x, y, p0, fit_x=None, bounds=(-np.inf, np.inf)):
        if fit_x is None:
            fit_x = x
        coeff, var_matrix = curve_fit(mapm_spe_fit, x, y, p0=p0, bounds=bounds)
        result = mapm_spe_fit(fit_x, *coeff)
        return result, coeff

    def _get_subfits(self):
        subfits = dict()
        if self.coeff:
            def pedestal_kw(x, norm, eped, eped_sigma, lambda_, **kw):
                return pedestal_signal(x, norm, eped, eped_sigma, lambda_)

            def pe_kw(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_, **kw):
                self.k = np.arange(1, 11)
                return pe_signal(self.k[:, None], x[None, :], norm, eped,
                                 eped_sigma, spe, spe_sigma, lambda_)

            pedestal = pedestal_kw(self.fit_x, **self.coeff)
            pe_fits = pe_kw(self.fit_x, **self.coeff)
            subfits = dict(pedestal=pedestal)
            for i, pe_i in enumerate(self.k):
                subfits['{}pe'.format(pe_i)] = pe_fits[i]
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
        self.initial = [None]*3
        self.bounds = ([-np.inf]*3, [np.inf]*3)

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
            fit, coeff = self._fit(hist, edges, between, fit_x)
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

    def _fit(self, hist, edges, between, fit_x):
        p0 = self.initial
        bounds = self.bounds
        if not p0[0]:
            p0[0] = hist.max()
        if not p0[1]:
            p0[1] = self._bright_mean
        if not p0[2]:
            p0[2] = self._bright_std
        fit, coeff = self.scipy_fit(between, hist, p0, fit_x, bounds)
        coeff_dict = dict()
        for i, c in enumerate(self.coeff_list):
            coeff_dict[c] = coeff[i]
        return fit, coeff_dict

    @staticmethod
    def scipy_fit(x, y, p0, fit_x=None, bounds=(-np.inf, np.inf)):
        def gaus(x_, norm, mean, stddev):
            return norm * normal.pdf(x_, mean, stddev)

        if fit_x is None:
            fit_x = x
        coeff, var_matrix = curve_fit(gaus, x, y, p0=p0, bounds=bounds)
        result = gaus(fit_x, *coeff)
        return result, coeff
