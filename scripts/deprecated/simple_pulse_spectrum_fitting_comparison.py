import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import poisson, norm as gaussian
from scipy.optimize import curve_fit
from os.path import join, exists
from os import makedirs
from functools import partial
from IPython import embed


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
    x
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


def scipy_fit(x, y, p0, fit_x=None):
    if fit_x is None:
        fit_x = x
    coeff, var_matrix = curve_fit(mapm_spe_fit, x, y, p0=p0)
    result = mapm_spe_fit(fit_x, *coeff)
    return result, coeff


def scipy_fit_multi_lambda(x_list, y_list, p0, fit_x=None):
    if fit_x is None:
        fit_x = x_list

    x_data = np.hstack(x_list)
    y_data = np.hstack(y_list)
    fit_x_data = np.hstack(fit_x)

    s = []
    s_plot = []
    l = 0
    for xi in x_list:
        c = xi.size + l
        s.append(np.s_[l:c])
        l = c
    l = 0
    for xi in fit_x:
        c = xi.size + l
        s_plot.append(np.s_[l:c])
        l = c

    def multi_fit(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_,
                  norm2, lambda_2, norm3, lambda_3, slice):
        x1 = x[slice[0]]
        x2 = x[slice[1]]
        x3 = x[slice[2]]

        fit = mapm_spe_fit
        multi = np.hstack(
            [fit(x1, norm, eped, eped_sigma, spe, spe_sigma, lambda_),
             fit(x2, norm2, eped, eped_sigma, spe, spe_sigma, lambda_2),
             fit(x3, norm3, eped, eped_sigma, spe, spe_sigma, lambda_3)])
        return multi

    mf_fitting = partial(multi_fit, slice=s)
    mf_plotting = partial(multi_fit, slice=s_plot)

    coeff_m, var_matrix = curve_fit(mf_fitting, x_data, y_data, p0=p0)
    scipy_y_result_multi_all = mf_plotting(fit_x_data, *coeff_m)

    result1 = scipy_y_result_multi_all[s_plot[0]]
    result2 = scipy_y_result_multi_all[s_plot[1]]
    result3 = scipy_y_result_multi_all[s_plot[2]]

    return result1, result2, result3, coeff_m


def plot_multi_lambda(between, hist, edges):
    """

    Parameters
    ----------
    between : list[3]
    hist : list[3]
    edges : list[3]

    Returns
    -------

    """

    # Setup fit x inputs
    fit_x = [np.linspace(edges[0][0], edges[0][-1], edges[0].size*10),
             np.linspace(edges[1][0], edges[1][-1], edges[1].size*10),
             np.linspace(edges[2][0], edges[2][-1], edges[2].size*10)]

    # Single Fit
    p0 = [200000, 0, 5, 20, 5, 1]
    fs1, cs1 = scipy_fit(between[0], hist[0], p0, fit_x[0])
    fs2, cs2 = scipy_fit(between[1], hist[1], p0, fit_x[1])
    fs3, cs3 = scipy_fit(between[2], hist[2], p0, fit_x[2])

    # Multi Fit
    p0 = [200000, 0, 5, 20, 5, 1, 200000, 1, 200000, 1]
    fm1, fm2, fm3, cm = scipy_fit_multi_lambda(between, hist, p0, fit_x)

    fig = plt.figure(figsize=(13, 6))
    fig.subplots_adjust(wspace=0.4, hspace=1)
    ax_m1 = plt.subplot2grid((3,3), (0,0), rowspan=2)
    ax_m1.set_title('Laser Setting 4.10')
    ax_m1.set_xlabel('Integrated ADC')
    ax_m2 = plt.subplot2grid((3,3), (0,1), rowspan=2)
    ax_m2.set_title('Laser Setting 4.05')
    ax_m2.set_xlabel('Integrated ADC')
    ax_m3 = plt.subplot2grid((3,3), (0,2), rowspan=2)
    ax_m3.set_title('Laser Setting 4.00')
    ax_m3.set_xlabel('Integrated ADC')
    ax_t1 = plt.subplot2grid((3,2), (2,0))
    ax_t1.set_title('Single Fit', y=1.2)
    ax_t2 = plt.subplot2grid((3,2), (2,1))
    ax_t2.set_title('Multi Fit', y=1.2)

    ax_m1.hist(between[0], edges[0], weights=hist[0],
               alpha=0.4, label="Spectrum")
    ax_m1.plot(fit_x[0], fs1, label="scipy_single")
    ax_m1.plot(fit_x[0], fm1, label="scipy_multi")
    ax_m1.legend(loc=1)

    ax_m2.hist(between[1], edges[1], weights=hist[1],
               alpha=0.4, label="Spectrum")
    ax_m2.plot(fit_x[1], fs2, label="scipy_single")
    ax_m2.plot(fit_x[1], fm2, label="scipy_multi")
    ax_m2.legend(loc=1)

    ax_m3.hist(between[2], edges[2], weights=hist[2],
               alpha=0.4, label="Spectrum")
    ax_m3.plot(fit_x[2], fs3, label="scipy_single")
    ax_m3.plot(fit_x[2], fm3, label="scipy_multi")
    ax_m3.legend(loc=1)

    table_data = [[*cs1], [*cs2], [*cs3]]
    table_data = [['%.3f' % j for j in i] for i in table_data]
    table_col = ("norm", "eped", "eped_sigma", "spe", "spe_sigma", "lambda")
    table_row = ("LS 4.10", "LS 4.05", "LS 4.00")
    # ax_t1.axis('tight')
    ax_t1.axis('off')
    table1 = ax_t1.table(cellText=table_data, colLabels=table_col,
                         rowLabels=table_row, loc='center')
    table1.scale(1.2, 2)
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)

    table_data = [[cm[0], cm[1], cm[2], cm[3], cm[4], cm[5]],
                  [cm[6], cm[1], cm[2], cm[3], cm[4], cm[7]],
                  [cm[8], cm[1], cm[2], cm[3], cm[4], cm[9]]]
    table_data = [['%.3f' % j for j in i] for i in table_data]
    table_col = ("norm", "eped", "eped_sigma", "spe", "spe_sigma", "lambda")
    table_row = ("LS 4.10", "LS 4.05", "LS 4.00")
    # ax_t2.axis('tight')
    ax_t2.axis('off')
    table2 = ax_t2.table(cellText=table_data, colLabels=table_col,
                         rowLabels=table_row, loc='center')
    table2.scale(1.2, 2)
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)

    return fig


def plot_mask_comparison(between, hist, edges):
    """

    Parameters
    ----------
    between : list[2]
    hist : list[2]
    edges : list[2]

    Returns
    -------

    """

    # Setup fit x inputs
    fit_x = [np.linspace(edges[0][0], edges[0][-1], edges[0].size*10),
             np.linspace(edges[1][0], edges[1][-1], edges[1].size*10)]

    # Single Fit
    p0 = [200000, 0, 5, 20, 5, 1]
    fs1, cs1 = scipy_fit(between[0], hist[0], p0, fit_x[0])
    fs2, cs2 = scipy_fit(between[1], hist[1], p0, fit_x[1])

    fig = plt.figure(figsize=(13, 6))
    fig.subplots_adjust(wspace=0.4, hspace=1)
    ax_m1 = plt.subplot2grid((3,2), (0,0), rowspan=2)
    ax_m1.set_title('Laser Setting 3.95, Masked Module')
    ax_m1.set_xlabel('Integrated ADC')
    ax_m2 = plt.subplot2grid((3,2), (0,1), rowspan=2)
    ax_m2.set_title('Laser Setting 3.95, Unmasked Module')
    ax_m2.set_xlabel('Integrated ADC')
    ax_t1 = plt.subplot2grid((3,2), (2,0), colspan=2)
    ax_t1.set_title('Single Fit', y=1.2)

    ax_m1.hist(between[0], edges[0], weights=hist[0],
               alpha=0.4, label="Spectrum")
    ax_m1.plot(fit_x[0], fs1, label="scipy_single")
    ax_m1.legend(loc=1)

    ax_m2.hist(between[1], edges[1], weights=hist[1],
               alpha=0.4, label="Spectrum")
    ax_m2.plot(fit_x[1], fs2, label="scipy_single")
    ax_m2.legend(loc=1)

    table_data = [[*cs1], [*cs2]]
    table_data = [['%.3f' % j for j in i] for i in table_data]
    table_col = ("norm", "eped", "eped_sigma", "spe", "spe_sigma", "lambda")
    table_row = ("LS 3.95, Masked", "LS 3.95, Unmasked")
    # ax_t1.axis('tight')
    ax_t1.axis('off')
    table1 = ax_t1.table(cellText=table_data, colLabels=table_col,
                         rowLabels=table_row, loc='center')
    table1.scale(1, 2)
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)

    return fig


def main():
    input_path1 = "/Users/Jason/Software/outputs/lab/MPIK_Lab_Comissioning/Run00180_r0/extract_pulse_spectrum/height_area.npz"
    file1 = np.load(input_path1)
    height1 = file1['height']
    area1 = file1['area']

    input_path2 = "/Users/Jason/Software/outputs/lab/MPIK_Lab_Comissioning/Run00181_r0/extract_pulse_spectrum/height_area.npz"
    file2 = np.load(input_path2)
    height2 = file2['height']
    area2 = file2['area']

    input_path3 = "/Users/Jason/Software/outputs/lab/MPIK_Lab_Comissioning/Run00182_r0/extract_pulse_spectrum/height_area.npz"
    file3 = np.load(input_path3)
    height3 = file3['height']
    area3 = file3['area']

    input_path4 = "/Users/Jason/Software/outputs/lab/MPIK_Lab_Comissioning/Run00183_r0/extract_pulse_spectrum/height_area.npz"
    file4 = np.load(input_path4)
    height4 = file4['height']
    area4 = file4['area']

    input_path5 = "/Users/Jason/Software/outputs/lab/MPIK_Lab_Comissioning/Run00184_r0/extract_pulse_spectrum/height_area.npz"
    file5 = np.load(input_path5)
    height5 = file5['height']
    area5 = file5['area']

    fig_dir = "/Users/Jason/Software/outputs/lab/MPIK_Lab_Comissioning/fits"
    if not exists(fig_dir):
        print("Creating directory: {}".format(fig_dir))
        makedirs(fig_dir)

    hist_h1_817, edges_h1_817 = np.histogram(height1[:, 817], bins=40, range=[-5, 15])
    between_h1_817 = (edges_h1_817[1:] + edges_h1_817[:-1]) / 2
    hist_a1_817, edges_a1_817 = np.histogram(area1[:, 817], bins=40, range=[-20, 60])
    between_a1_817 = (edges_a1_817[1:] + edges_a1_817[:-1]) / 2

    hist_h2_817, edges_h2_817 = np.histogram(height2[:, 817], bins=40, range=[-5, 15])
    between_h2_817 = (edges_h2_817[1:] + edges_h2_817[:-1]) / 2
    hist_a2_817, edges_a2_817 = np.histogram(area2[:, 817], bins=40, range=[-20, 60])
    between_a2_817 = (edges_a2_817[1:] + edges_a2_817[:-1]) / 2

    hist_h3_817, edges_h3_817 = np.histogram(height3[:, 817], bins=40, range=[-5, 15])
    between_h3_817 = (edges_h3_817[1:] + edges_h3_817[:-1]) / 2
    hist_a3_817, edges_a3_817 = np.histogram(area3[:, 817], bins=40, range=[-20, 60])
    between_a3_817 = (edges_a3_817[1:] + edges_a3_817[:-1]) / 2

    hist_h4_817, edges_h4_817 = np.histogram(height4[:, 817], bins=40, range=[-5, 15])
    between_h4_817 = (edges_h4_817[1:] + edges_h4_817[:-1]) / 2
    hist_a4_817, edges_a4_817 = np.histogram(area4[:, 817], bins=40, range=[-20, 60])
    between_a4_817 = (edges_a4_817[1:] + edges_a4_817[:-1]) / 2

    hist_h5_817, edges_h5_817 = np.histogram(height5[:, 817], bins=40, range=[-5, 15])
    between_h5_817 = (edges_h5_817[1:] + edges_h5_817[:-1]) / 2
    hist_a5_817, edges_a5_817 = np.histogram(area5[:, 817], bins=40, range=[-20, 60])
    between_a5_817 = (edges_a5_817[1:] + edges_a5_817[:-1]) / 2

    hist_h1_2, edges_h1_2 = np.histogram(height1[:, 2], bins=40, range=[-5, 15])
    between_h1_2 = (edges_h1_2[1:] + edges_h1_2[:-1]) / 2
    hist_a1_2, edges_a1_2 = np.histogram(area1[:, 2], bins=40, range=[-20, 60])
    between_a1_2 = (edges_a1_2[1:] + edges_a1_2[:-1]) / 2

    hist_h2_2, edges_h2_2 = np.histogram(height2[:, 2], bins=40, range=[-5, 15])
    between_h2_2 = (edges_h2_2[1:] + edges_h2_2[:-1]) / 2
    hist_a2_2, edges_a2_2 = np.histogram(area2[:, 2], bins=40, range=[-20, 60])
    between_a2_2 = (edges_a2_2[1:] + edges_a2_2[:-1]) / 2

    hist_h3_2, edges_h3_2 = np.histogram(height3[:, 2], bins=40, range=[-5, 15])
    between_h3_2 = (edges_h3_2[1:] + edges_h3_2[:-1]) / 2
    hist_a3_2, edges_a3_2 = np.histogram(area3[:, 2], bins=40, range=[-20, 60])
    between_a3_2= (edges_a3_2[1:] + edges_a3_2[:-1]) / 2

    between_817 = [between_a1_817, between_a2_817, between_a3_817]
    hist_817 = [hist_a1_817, hist_a2_817, hist_a3_817]
    edges_817 = [edges_a1_817, edges_a2_817, edges_a3_817]
    fig_a_817_lambda = plot_multi_lambda(between_817, hist_817, edges_817)
    fig_a_817_lambda.suptitle("Pulse Area Spectrum, Pixel 817, Masked Module (1 Unmasked Pixel)")
    fig_a_817_lambda.savefig(join(fig_dir, "multi_lambda_area_masked.pdf"))

    between_2 = [between_a1_2, between_a2_2, between_a3_2]
    hist_2 = [hist_a1_2, hist_a2_2, hist_a3_2]
    edges_2 = [edges_a1_2, edges_a2_2, edges_a3_2]
    fig_a_2_lambda = plot_multi_lambda(between_2, hist_2, edges_2)
    fig_a_2_lambda.suptitle("Pulse Area Spectrum, Pixel 2, Unmasked Module")
    fig_a_2_lambda.savefig(join(fig_dir, "multi_lambda_area_unmasked.pdf"))

    between_m = [between_a4_817, between_a5_817]
    hist_m = [hist_a4_817, hist_a5_817]
    edges_m = [edges_a4_817, edges_a5_817]
    fig_a_817_m = plot_mask_comparison(between_817, hist_817, edges_817)
    fig_a_817_m.suptitle("Pulse Area Spectrum, Pixel 817, Masked Vs. Unmasked")
    fig_a_817_m.savefig(join(fig_dir, "area_masked_vs_unmasked.pdf"))


if __name__ == '__main__':
    main()
