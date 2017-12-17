import argparse
from os.path import join, exists
from os import makedirs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange
from fitter import CHECSSPEFitter as Fitter
from copy import deepcopy
from IPython import embed


def get_fitter():
    range_ = [-50, 250]
    nbins = 100
    initial = dict(
        norm=None,  # Automatically calculated by fitter from histogram
        eped=-10,
        eped_sigma=4,
        spe=37,
        spe_sigma=6,
        lambda_=4,
        opct=0.4,
        pap=0.3,
        dap=0.5
    )
    limit = dict(
        limit_norm=(0, 100000),
        limit_eped=(-20, 10),
        limit_eped_sigma=(2, 10),
        limit_spe=(20, 40),
        limit_spe_sigma=(2, 10),
        limit_lambda_=(2, 5),
        limit_opct=(0, 0.8),
        limit_pap=(0, 0.8),
        limit_dap=(0, 0.8)
    )
    fix = dict(
        fix_norm=True,
    )
    fitter = Fitter()
    fitter.range = range_
    fitter.nbins = nbins
    fitter.initial = initial
    fitter.limits = limit
    fitter.fix = fix
    return fitter


def main():
    description = 'SPE fit a single illumination'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-f', '--file1', dest='file', action='store',
                        required=True, help='lowest light level npy filepath')
    parser.add_argument('-p', '--pixel', dest='pixel', action='store',
                        required=True, help='pixel to investigate', type=int)
    args = parser.parse_args()

    charge = np.load(args.file)

    poi = args.pixel

    fitter = get_fitter()
    fitter_poi = None

    df_list = []
    n_pixels = charge.shape[1]
    desc = "Fitting SPE spectrum of each pixel"
    for p in trange(n_pixels, desc=desc):
        if p != poi: continue  # TEMP while fitting parameters are revised

        fitter.apply(charge[:, p])

        d = fitter.coeff.copy()
        d['pixel'] = p

        df_list.append(d)

        if p == poi:
            fitter_poi = deepcopy(fitter)

    df = pd.DataFrame(df_list)

    output_dir = "spe_single_fit"
    if not exists(output_dir):
        print("Creating directory: {}".format(output_dir))
        makedirs(output_dir)

    fig_fit = plt.figure(figsize=(13, 6))
    fig_fit.suptitle("Single Fit")
    ax = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
    ax_t = plt.subplot2grid((3, 2), (0, 1), rowspan=3)
    x = np.linspace(fitter_poi.range[0], fitter_poi.range[1], 1000)
    hist = fitter_poi.hist
    edges = fitter_poi.edges
    between = fitter_poi.between
    coeff = fitter_poi.coeff.copy()
    coeffl = fitter_poi.coeff_list.copy()
    initial = fitter_poi.p0.copy()
    fit = fitter_poi.fit_function(x, **coeff)
    init = fitter_poi.fit_function(x, **initial)
    rc2 = fitter_poi.reduced_chi2
    pval = fitter_poi.p_value
    ax.hist(between, bins=edges, weights=hist, histtype='step', label="Hist")
    ax.plot(x, fit, label="Fit")
    ax.plot(x, init, label="Initial")
    ax.legend(loc=1, frameon=True, fancybox=True, framealpha=0.7)
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
