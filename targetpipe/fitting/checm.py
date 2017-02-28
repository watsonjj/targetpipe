import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm as normal
from ctapipe.core import Component
from targetpipe.fitting.mapm_spe import mapm_spe_fit, pedestal_signal, pe_signal
from traitlets import CaselessStrEnum as CaStEn, Bool


class CHECMFitter(Component):
    name = 'CHECMFitter'

    possible = ['spe', 'bright']
    brightness = CaStEn(possible, 'spe',
                        help='Brightness of run').tag(config=True)
    apply_fit = Bool(True, help='Apply the fit').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        Generic fitter for MAPMs

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

        if self.brightness == 'spe':
            self.get_histogram = self.get_histogram_spe
            self._fit = self._fit_spe
            self._get_gain = self._get_gain_spe
            self._get_subfits = self._get_spe_subfits
            self.subfit_labels = ['pedestal']
            for pe in self.k:
                self.subfit_labels.append('{}pe'.format(pe))
        elif self.brightness == 'bright':
            self.get_histogram = self.get_histogram_bright
            self._fit = self._fit_bright
            self._get_gain = self._get_gain_bright
            self._get_subfits = self._get_none_subfits
        else:
            self.log.error("Unknown brightness setting: {}"
                           .format(self.brightness))

        if not self.apply_fit:
            self._fit = self._fit_none
            self._get_gain = self._get_gain_none

    @property
    def gain(self):
        return self._get_gain()

    @property
    def subfits(self):
        return self._get_subfits()

    def apply(self, spectrum, height=False):
        hist, edges = self.get_histogram(spectrum, height)
        between = (edges[1:] + edges[:-1]) / 2

        fit, coeff, fit_x = self._fit(hist, edges, between)

        self.hist = hist
        self.edges = edges
        self.between = between
        self.fit = fit
        self.coeff = coeff
        self.fit_x = fit_x

    @staticmethod
    def get_histogram_spe(spectrum, height=False):
        if height:
            hist, edges = np.histogram(spectrum, bins=40, range=[-5, 15])
        else:
            hist, edges = np.histogram(spectrum, bins=40, range=[-20, 60])
        return hist, edges

    def get_histogram_bright(self, spectrum, height=False):
        self._bright_mean = np.mean(spectrum)
        self._bright_std = np.std(spectrum)
        range_ = [self._bright_mean - 3 * self._bright_std,
                  self._bright_mean + 3 * self._bright_std]
        hist, edges = np.histogram(spectrum, bins=40, range=range_)
        return hist, edges

    def _get_gain_spe(self):
        return self.coeff['spe']

    def _get_gain_bright(self):
        return self.coeff['mean']

    @staticmethod
    def _get_gain_none():
        return 0

    def _fit_spe(self, hist, edges, between):
        p0 = [200000, 0, 5, 20, 5, 1]
        fit_x = np.linspace(edges[0], edges[-1], edges.size*10)
        fit, coeff = self.scipy_fit_spe(between, hist, p0, fit_x)
        coeff_dict = dict(norm=coeff[0], eped=coeff[1], eped_sigma=coeff[2],
                          spe=coeff[3], spe_sigma=coeff[4], lambda_=coeff[5])
        return fit, coeff_dict, fit_x

    def _fit_bright(self, hist, edges, between):
        p0 = [hist.max(), self._bright_mean, self._bright_std]
        fit_x = np.linspace(edges[0], edges[-1], edges.size*10)
        fit, coeff = self.scipy_fit_bright(between, hist, p0, fit_x)
        coeff_dict = dict(norm=coeff[0], mean=coeff[1], stddev=coeff[2])
        return fit, coeff_dict, fit_x

    @staticmethod
    def _fit_none(hist, edges, between):
        fit = np.zeros(between.shape)
        fit_x = np.arange(between.size)
        coeff = None
        return fit, coeff, fit_x

    @staticmethod
    def scipy_fit_spe(x, y, p0, fit_x=None):
        if fit_x is None:
            fit_x = x
        coeff, var_matrix = curve_fit(mapm_spe_fit, x, y, p0=p0)
        result = mapm_spe_fit(fit_x, *coeff)
        return result, coeff

    @staticmethod
    def scipy_fit_bright(x, y, p0, fit_x=None):
        def gaus(x_, norm, mean, stddev):
            return norm * normal.pdf(x_, mean, stddev)

        if fit_x is None:
            fit_x = x
        coeff, var_matrix = curve_fit(gaus, x, y, p0=p0)
        result = gaus(fit_x, *coeff)
        return result, coeff

    def _get_spe_subfits(self):
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

    def _get_none_subfits(self):
        return dict()