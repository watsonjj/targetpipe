import numpy as np
from scipy.stats import poisson, norm as gaussian
from scipy.optimize import curve_fit


def pedestal_signal(x, norm, eped, eped_sigma, lambda_):
    """
    Obtain the signal provided by the pedestal in the pulse spectrum.

    Parameters
    ----------
    x : 1darray
        The x values to evaluate at
    norm : float
        Integral of the zeroth peak in the distribution, represents p(0)
    eped : float
        Distance of the zeroth peak from the origin
    eped_sigma : float
        Sigma of the zeroth peak, represents electronic noise of the system
    lambda_ : float
        Poisson mean

    Returns
    -------
    signal : ndarray
        The y values of the signal provided by the pedestal.

    """
    p_ped = poisson.pmf(0, lambda_)
    signal = norm * p_ped * gaussian.pdf(x, eped, eped_sigma) / eped_sigma
    return signal


def pe_signal(k, x, norm, eped, eped_sigma, spe, spe_sigma, lambda_):
    """
    Obtain the signal provided by photoelectrons in the pulse spectrum.

    Parameters
    ----------
    k : int or 1darray
        The NPEs to evaluate. A list of NPEs can be passed here, provided it
        is broadcast as [:, None], and the x input is broadcast as [None, :],
        the return value will then be a shape [k.size, x.size].

        k must be greater than or equal to 1.
    x : 1darray
        The x values to evaluate at
    norm : float
        Integral of the zeroth peak in the distribution, represents p(0)
    eped : float
        Distance of the zeroth peak from the origin
    eped_sigma : float
        Sigma of the zeroth peak, represents electronic noise of the system
    spe : float
        Signal produced by 1 photo-electron
    spe_sigma : float
        Spread in the number of photo-electrons incident on the MAPMT
    lambda_ : float
        Poisson mean

    Returns
    -------
    signal : ndarray
        The y values of the signal provided by the photoelectrons. If k is an
        integer, this will have same shape as x. If k is an array,
        and k and x are broadcase correctly, this will have
        shape [k.size, x.size].

    """
    # Obtain poisson distribution
    # TODO: could do something smarter here depending on lambda_
    p = poisson.pmf(np.arange(11), lambda_)

    pe = eped + k * spe
    pe_sigma = np.sqrt(k * spe_sigma ** 2 + eped_sigma ** 2)
    signal = norm * p[k] * gaussian.pdf(x, pe, pe_sigma) / pe_sigma
    return signal


def mapm_spe_fit(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_):
    """
    Fit for the SPE spectrum of a MAPM

    Parameters
    ----------
    x : 1darray
        The x values to evaluate at
    norm : float
        Integral of the zeroth peak in the distribution, represents p(0)
    eped : float
        Distance of the zeroth peak from the origin
    eped_sigma : float
        Sigma of the zeroth peak, represents electronic noise of the system
    spe : float
        Signal produced by 1 photo-electron
    spe_sigma : float
        Spread in the number of photo-electrons incident on the MAPMT
    lambda_ : float
        Poisson mean

    Returns
    -------

    """

    # Obtain pedestal signal
    params = [norm, eped, eped_sigma, lambda_]
    ped_s = pedestal_signal(x, *params)

    # Obtain pe signal
    k = np.arange(1, 11)
    params = [norm, eped, eped_sigma, spe, spe_sigma, lambda_]
    pe_s = pe_signal(k[:, None], x[None, :], *params).sum(0)

    signal = ped_s + pe_s

    return signal
