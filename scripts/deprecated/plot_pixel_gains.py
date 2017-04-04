import numpy as np
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure, output_file, show
from scipy.optimize import curve_fit
from os.path import join, exists, dirname, splitext, basename
from os import makedirs
from functools import partial
from targetpipe.fitting.mapm_spe import mapm_spe_fit
from tqdm import tqdm
from IPython import embed


def get_gain(x, y, p0):
    """
    Fit the MAPM SPE using scipy's curve_fit.

    Parameters
    ----------
    x : 1darray
        The x values to fit at
    y : 1darray
        The y values to fit to
    p0 : list[6]
        Initial values for the coefficients for the fit

    Returns
    -------
    gain : float
        The

    """
    coeff, var_matrix = curve_fit(mapm_spe_fit, x, y, p0=p0)
    gain = coeff[3]
    sigma = coeff[4]
    return gain, sigma


def main():
    input_path1 = "/Users/Jason/Mounts/mpik_runs/Run00426_r0/extract_pulse_spectrum/height_area.npz"
    file1 = np.load(input_path1)
    height1 = file1['height']
    area1 = file1['area']

    p0 = [200000, 0, 5, 20, 5, 1]

    n_events, n_pix = area1.shape
    gain = np.zeros(n_pix)
    sigmas = np.zeros(n_pix)

    desc = "Fitting Pixels"
    with tqdm(total=n_pix, desc=desc) as pbar:
        for pix in range(n_pix):
            pbar.update(1)

            hist, edges = np.histogram(area1[:, pix], bins=40, range=[-20, 60])
            between = (edges[1:] + edges[:-1]) / 2

            try:
                gain[pix], sigmas[pix] = get_gain(between, hist, p0)
            except RuntimeError:
                print("Pixel {} could not be fitted".format(pix))

    x = np.arange(n_pix)
    stddev = np.sqrt(sigmas)

    fig_dir = join(dirname(input_path1), splitext(basename(input_path1))[0],
                   "plot_pixel_gains")
    if not exists(fig_dir):
        print("Creating directory: {}".format(fig_dir))
        makedirs(fig_dir)

    cdsource_d = dict(x=x, gain=gain, stddev=stddev)
    cdsource = ColumnDataSource(data=cdsource_d)

    output_file(join(fig_dir, 'gain_vs_pixel.html'))
    tools = "xpan, xwheel_pan, box_zoom, xwheel_zoom, save, reset"
    fig = figure(plot_width=1000, plot_height=700, tools = tools,
                 active_scroll='xwheel_zoom', title='Pixel Gains')
    fig.xaxis.axis_label = "Pixel"
    fig.yaxis.axis_label = "Integrated ADC"
    c1 = fig.circle(source=cdsource, x='x', y='gain')
    fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @gain)"),
                                      ("stddev", "@stddev")], renderers=[c1]))

    top = gain + stddev
    bottom = gain - stddev
    left = x - 0.3
    right = x + 0.3
    fig.segment(x0=x, y0=bottom, x1=x, y1=top,
                line_width=1.5, color='black')
    fig.segment(x0=left, y0=top, x1=right, y1=top,
                line_width=1.5, color='black')
    fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
                line_width=1.5, color='black')

    show(fig)


if __name__ == '__main__':
    main()
