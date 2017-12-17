import argparse
from os.path import join, exists
from os import makedirs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange
from fitter import CHECSSPEMultiFitter as Fitter
from copy import deepcopy
from IPython import embed


def get_fitter():
    range_ = [-50, 250]
    nbins = 100
    initial = dict(
        norm1=None,  # Automatically calculated by fitter from histogram
        norm2=None,  # Automatically calculated by fitter from histogram
        norm3=None,  # Automatically calculated by fitter from histogram
        eped=0,
        eped_sigma=4,
        spe=40,
        spe_sigma=6,
        lambda_1=4,
        lambda_2=4,
        lambda_3=4,
        opct=0.4,
        pap=0.3,
        dap=0.5
    )
    limit = dict(
        limit_norm1=(0, 100000),
        limit_norm2=(0, 100000),
        limit_norm3=(0, 100000),
        limit_eped=(-10, 10),
        limit_eped_sigma=(2, 10),
        limit_spe=(30, 50),
        limit_spe_sigma=(2, 10),
        limit_lambda_1=(2, 5),
        limit_lambda_2=(2, 5),
        limit_lambda_3=(2, 5),
        limit_opct=(0, 0.8),
        limit_pap=(0, 0.8),
        limit_dap=(0, 0.8)
    )
    fix = dict(
        fix_norm1=True,
        fix_norm2=True,
        fix_norm3=True,
    )
    fitter = Fitter()
    fitter.range = range_
    fitter.nbins = nbins
    fitter.initial = initial
    fitter.limits = limit
    fitter.fix = fix
    return fitter


def main():
    description = 'SPE fit multiple illuminations simultaneously'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-f1', '--file1', dest='file1', action='store',
                        required=True, help='lowest light level npy filepath')
    parser.add_argument('-f2', '--file2', dest='file2', action='store',
                        required=True, help='medium light level npy filepath')
    parser.add_argument('-f3', '--file3', dest='file3', action='store',
                        required=True, help='highest light level npy filepath')
    parser.add_argument('-p', '--pixel', dest='pixel', action='store',
                        required=True, help='pixel to investigate', type=int)
    args = parser.parse_args()

    charge1 = np.load(args.file1)
    charge2 = np.load(args.file2)
    charge3 = np.load(args.file3)

    poi = args.pixel

    fitter = get_fitter()
    fitter_poi = None

    df_list = []
    n_pixels = charge1.shape[1]
    desc = "Fitting SPE spectrum of each pixel"
    for p in trange(n_pixels, desc=desc):
        if p != poi: continue  # TEMP while fitting parameters are revised

        fitter.apply_multi(charge1[:, p], charge2[:, p], charge3[:, p])

        d = fitter.coeff.copy()
        d['pixel'] = p

        df_list.append(d)

        if p == poi:
            fitter_poi = deepcopy(fitter)

    df = pd.DataFrame(df_list)

    output_dir = "spe_multi_fit"
    if not exists(output_dir):
        print("Creating directory: {}".format(output_dir))
        makedirs(output_dir)

    fig_fit = plt.figure(figsize=(13, 6))
    fig_fit.suptitle("Multi Fit")
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax3 = plt.subplot2grid((3, 2), (2, 0))
    ax_t = plt.subplot2grid((3, 2), (0, 1), rowspan=3)
    x = np.linspace(fitter_poi.range[0], fitter_poi.range[1], 1000)
    hist1, hist2, hist3 = fitter_poi.hist
    edges = fitter_poi.edges
    between = fitter_poi.between
    coeff = fitter_poi.coeff.copy()
    coeffl = fitter_poi.coeff_list.copy()
    initial = fitter_poi.p0.copy()
    fit1, fit2, fit3 = fitter_poi.fit_function(x, **coeff)
    init1, init2, init3 = fitter_poi.fit_function(x, **initial)
    rc2 = fitter_poi.reduced_chi2
    pval = fitter_poi.p_value
    ax1.hist(between, bins=edges, weights=hist1, histtype='step', label="Hist")
    ax1.plot(x, fit1, label="Fit")
    ax1.plot(x, init1, label="Initial")
    ax1.legend(loc=1, frameon=True, fancybox=True, framealpha=0.7)
    ax2.hist(between, bins=edges, weights=hist2, histtype='step', label="Hist")
    ax2.plot(x, fit2, label="Fit")
    ax2.plot(x, init2, label="Initial")
    ax3.hist(between, bins=edges, weights=hist3, histtype='step', label="Hist")
    ax3.plot(x, fit3, label="Fit")
    ax3.plot(x, init3, label="Initial")
    ax_t.axis('off')
    td = [[initial[i], '%.3f' % coeff[i]] for i in coeffl]
    td.append(["", '%.3g' % rc2])
    td.append(["", '%.3g' % pval])
    tr = coeffl
    tr.append("Reduced Chi^2")
    tr.append("P-Value")
    tc = ['Initial', 'Fit']
    table = ax_t.table(cellText=td, rowLabels=tr, colLabels=tc, loc='center')
    table.set_fontsize(6)
    output_path = join(output_dir, "fit_p{}.pdf".format(poi))
    fig_fit.savefig(output_path, bbox_inches='tight')
    print("Figure saved to: {}".format(output_path))


if __name__ == '__main__':
    main()
