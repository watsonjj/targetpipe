import numpy as np
from scipy.stats import poisson, norm as gaussian
from scipy.special import binom


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
    signal = norm * p_ped * gaussian.pdf(x, eped, eped_sigma)
    return signal


def pe_signal(k, x, norm, eped, eped_sigma, spe, spe_sigma, lambda_, opct, pap, dap):
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
    gnorm = 1#0.65#3.0/math.sqrt(2*3.1415926)
    npeaks = 11

    # print(norm, eped, eped_sigma, spe, spe_sigma, lambda_, opct, pap, dap)

    n = np.arange(npeaks)[:, None]
    j = np.arange(npeaks)[None, :]
    pj = poisson.pmf(j, lambda_)
    pct = np.sum(pj * np.power(1-opct, j) * np.power(opct, n - j) * binom(n-1, j-1), 1)

    sap = spe_sigma
    d1ap = dap
    d2ap = 2*dap
    if d1ap > d2ap:
        d1ap = 2*dap
        d2ap = dap

    # pap_ap1 = pct * pap
    # pap_ap2 = pct * pap ** 2
    # pap_noap = pct - pap_ap1 - pap_ap2

    papk = np.power(1 - pap, n[:, 0])
    p0AP = pct * papk
    pAP1 = pct * (1-papk) * papk
    pAP2 = pct * (1-papk) * (1-papk)

    pe_sigma = np.sqrt(k * spe_sigma ** 2 + eped_sigma ** 2)
    ap_sigma = np.sqrt(k * sap ** 2 + eped_sigma ** 2)

    signal = p0AP[k] * gaussian.pdf(x, eped + k * spe, pe_sigma)
    signal += pAP1[k] * gaussian.pdf(x, eped + k * spe * (1.0-d1ap), ap_sigma)
    signal += pAP2[k] * gaussian.pdf(x, eped + k * spe * (1.0-d2ap), ap_sigma)

    signal *= gnorm * norm

    return signal


def sipm_spe_fit(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_, opct, pap, dap):
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
    params = [norm, eped, eped_sigma, spe, spe_sigma, lambda_, opct, pap, dap]
    pe_s = pe_signal(k[:, None], x[None, :], *params).sum(0)

    signal = ped_s + pe_s

    return signal
