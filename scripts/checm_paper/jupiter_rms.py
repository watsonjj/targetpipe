import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from traitlets import Dict, List
from tqdm import tqdm
from ctapipe.core import Tool, Component
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.image.charge_extractors import NeighbourPeakIntegrator
from ctapipe.image.waveform_cleaning import CHECMWaveformCleanerLocal
from ctapipe.image import tailcuts_clean
from ctapipe.image.hillas import hillas_parameters
from ctapipe.instrument import CameraGeometry
from ctapipe.instrument.camera import _get_min_pixel_seperation
from ctapipe.visualization import CameraDisplay
from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.calib.camera.r1 import TargetioR1Calibrator
from targetpipe.plots.official import OfficialPlotter
from targetpipe.io.pixels import get_geometry, checm_pixel_pos
from IPython import embed
from targetpipe.plots.quick_camera import plot_quick_camera


class ImagePlotter(OfficialPlotter):
    name = 'ImagePlotter'

    def __init__(self, config, tool, **kwargs):
        """
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

        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(1, 1, 1)

    def create(self, image, geom, title):
        camera = CameraDisplay(geom, ax=self.ax,
                               image=image,
                               cmap='viridis')
        camera.add_colorbar()
        camera.colorbar.set_label("Residual RMS (p.e.)")
        camera.image = image

        self.fig.suptitle("Jupiter RMS ON-OFF")
        self.ax.set_title(title)
        self.ax.axis('off')


class Gaussian2DFitter(Component):
    name = "Gaussian2DFitter"

    def __init__(self, config, tool, **kwargs):
        """
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

    @staticmethod
    def _fit_function(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        x, y = xy
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(
            theta) ** 2) / (2 * sigma_y ** 2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(
            2 * theta)) / (4 * sigma_y ** 2)
        c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(
            theta) ** 2) / (2 * sigma_y ** 2)
        g = offset + amplitude * np.exp(
            - (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo)
               + c * ((y - yo) ** 2)))
        return g.ravel()

    def fit(self, x, y, data):
        minsep = _get_min_pixel_seperation(x, y)
        p0 = (data.max(), x.mean(), y.mean(),
              minsep, minsep, 0, np.median(data))
        bounds = ([0, x.min(), y.min(), 0, 0, -np.inf, -np.inf],
                  [np.inf, x.max(), y.max(), np.inf, np.inf, np.inf, np.inf])
        popt, pcov = optimize.curve_fit(self._fit_function, (x, y), data,
                                        p0=p0, bounds=bounds)
        coeff = popt

        return coeff

    def get_curve(self, x, y, coeff):
        curve = self._fit_function((x, y), *coeff)
        curve = np.reshape(curve, x.shape)
        return curve


class Image3DPlotter(OfficialPlotter):
    name = 'ImagePlotter'

    def __init__(self, config, tool, **kwargs):
        """
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

        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.gca(projection='3d')

    def create(self, x, y, data, coeff, fitter):
        xu = np.unique(x)
        yu = np.unique(y)[::-1]
        xx, yy = np.meshgrid(xu, yu)
        xi = np.linspace(x.min(), x.max(), 200)
        yi = np.linspace(y.min(), y.max(), 200)[::-1]
        xxi, yyi = np.meshgrid(xi, yi)

        di = griddata((x, y), data, (xi[None, :], yi[:, None]), method='cubic')
        di = np.ma.masked_invalid(di)
        di = np.ma.filled(di, 0)
        df = fitter.get_curve(xxi, yyi, coeff)

        self.ax.plot_surface(xxi, yyi, di, rstride=1, cstride=1, cmap='Greys', alpha=0.2, linewidth=0, antialiased=False)
        self.ax.plot_surface(xxi, yyi, df, rstride=1, cstride=1, cmap='viridis', linewidth=0, antialiased=False)

        self.fig.suptitle("Fit Comparison")
        # self.ax.axis('off')
        self.ax.set_xlabel("X (degrees)")
        self.ax.set_ylabel("Y (degrees)")
        self.ax.set_zlabel("Residual RMS (p.e.)")


class JupiterRMS(Tool):
    name = "JupiterRMS"
    description = "Plot the RMS across the camera for the Jupiter observations"

    aliases = Dict(dict())
    classes = List([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        file_df_list = [
            dict(type="on", deg=0, mirrors=2, fi=1, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05451_rms.npy"),
            dict(type="on", deg=0, mirrors=2, fi=2, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05452_rms.npy"),
            dict(type="off", deg=0, mirrors=2, fi=1, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05453_rms.npy"),
            dict(type="off", deg=0, mirrors=2, fi=2, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05454_rms.npy"),
            dict(type="on", deg=2, mirrors=2, fi=1, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05455_rms.npy"),
            dict(type="on", deg=2, mirrors=2, fi=2, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05456_rms.npy"),
            dict(type="on", deg=3.5, mirrors=2, fi=1, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05457_rms.npy"),
            dict(type="on", deg=3.5, mirrors=2, fi=2, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05458_rms.npy"),
            dict(type="on", deg=0, mirrors=1, fi=1, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05459_rms.npy"),
            dict(type="on", deg=0, mirrors=1, fi=2, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05460_rms.npy"),
            dict(type="on", deg=0, mirrors=1, fi=3, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05461_rms.npy"),
            dict(type="on", deg=0, mirrors=1, fi=4, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05462_rms.npy"),
            dict(type="off", deg=0, mirrors=1, fi=1, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05463_rms.npy"),
            dict(type="off", deg=0, mirrors=1, fi=2, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05464_rms.npy"),
            dict(type="off", deg=0, mirrors=1, fi=3, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05465_rms.npy"),
            dict(type="off", deg=0, mirrors=1, fi=4, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05466_rms.npy"),
            dict(type="on", deg=2, mirrors=1, fi=1, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05471_rms.npy"),
            dict(type="on", deg=2, mirrors=1, fi=2, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05472_rms.npy"),
            dict(type="on", deg=2, mirrors=1, fi=3, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05473_rms.npy"),
            dict(type="on", deg=2, mirrors=1, fi=4, path="/Volumes/gct-jason/data/170330/onsky-jupiter/Run05474_rms.npy")
        ]
        self.df = pd.DataFrame(file_df_list)


    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"

    def start(self):
        kwargs = dict(config=self.config, tool=self)
        p_kwargs = kwargs
        p_kwargs['script'] = "checm_paper_jupiter_rms"

        def get_run_averages(x, pbar):
            pbar.update(1)
            array = np.load(x)
            return array.mean(0), array.shape[0]

        def get_mode_average(x):
            return np.sum(x['rms_pp'] * x['n_events']) / np.sum(x['n_events'])

        # n_files = len(self.df.index)
        # desc = "Opening numpy files"
        # with tqdm(total=n_files, desc=desc) as pbar:
        #     result = self.df['path'].apply(get_run_averages, pbar=pbar)
        #     self.df['rms_pp'], self.df['n_events'] = zip(*result)
        #
        # store = pd.HDFStore('/Users/Jason/Downloads/jupiter_rms.h5')
        # store['df'] = self.df

        store = pd.HDFStore('/Users/Jason/Downloads/jupiter_rms.h5')
        self.df = store['df']

        group = self.df.groupby(['mirrors', 'type', 'deg'])
        result = group.apply(get_mode_average)
        result = result[:, 'on', :] - result[:, 'off', 0.0]

        geom = get_geometry()
        for (mirrors, deg), image in result.items():
            title = "{}mirror_{}deg".format(mirrors, deg)
            p_kwargs['figure_name'] = title
            im = ImagePlotter(**p_kwargs)
            im.create(image, geom, title)
            im.save()

        # Investigation: 1 mirror, 0.0 deg
        data = result[1, 0.0]
        p_kwargs['figure_name'] = "1mirror_0.0deg_fit"
        fitter = Gaussian2DFitter(**kwargs)
        x, y = checm_pixel_pos * 1/40.344e-3
        coeff = fitter.fit(x, y, data)
        fit_pix = fitter.get_curve(x, y, coeff)
        im = ImagePlotter(**p_kwargs)
        im.create(fit_pix, geom, "1mirror_0.0deg - 2D Gaussian Fit")
        im.save()
        p_kwargs['figure_name'] = "3d_comparison"
        im3d = Image3DPlotter(**p_kwargs)
        im3d.create(x, y, data, coeff, fitter)
        im3d.save()

    def finish(self):
        pass


from scipy import optimize
from scipy.interpolate import griddata

# def gaussian(height, center_x, center_y, width_x, width_y):
#     """Returns a gaussian function with the given parameters"""
#     width_x = float(width_x)
#     width_y = float(width_y)
#     return lambda x,y: height*np.exp(
#                 -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
#
# def moments(data, x_pos, y_pos):
#     """Returns (height, x, y, width_x, width_y)
#     the gaussian parameters of a 2D distribution by calculating its
#     moments """
#     total = data.sum()
#     X, Y = np.indices(data.shape)
#     x = (X*data).sum()/total
#     y = (Y*data).sum()/total
#     col = data[:, int(y)]
#     width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
#     row = data[int(x), :]
#     width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
#     height = data.max()
#
#     embed()
#
#     return height, x, y, width_x, width_y
#
# def fitgaussian(data, x_pos, y_pos):
#     """Returns (height, x, y, width_x, width_y)
#     the gaussian parameters of a 2D distribution found by a fit"""
#     params = moments(data, x_pos, y_pos)
#     errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
#                                  data)
#     p, success = optimize.leastsq(errorfunction, params)
#     return p
#
# # Create the gaussian data
# Xin, Yin = np.mgrid[0:201, 0:201]
# data = gaussian(3, 100, 100, 20, 40)(Xin, Yin) + np.random.random(Xin.shape)
#
# plt.matshow(data, cmap=plt.cm.gist_earth_r)
#
# params = fitgaussian(data)
# fit = gaussian(*params)
#
# plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
# ax = plt.gca()
# (height, x, y, width_x, width_y) = params
#
# plt.text(0.95, 0.05, """
# x : %.1f
# y : %.1f
# width_x : %.1f
# width_y : %.1f""" %(x, y, width_x, width_y),
#         fontsize=16, horizontalalignment='right',
#         verticalalignment='bottom', transform=ax.transAxes)


# def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
#     x, y = xy
#     xo = float(xo)
#     yo = float(yo)
#     a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
#     b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
#     c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
#     g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
#                             + c*((y-yo)**2)))
#     return g.ravel()
#
# # Create x and y indices
# x, y = checm_pixel_pos
# xu = np.unique(x)
# yu = np.unique(y)
# xx, yy = np.meshgrid(xu, yu[::-1])
# xi = np.linspace(x.min(), x.max(), 200)
# yi = np.linspace(y.min(), y.max(), 200)
# xxi, yyi = np.meshgrid(xi, yi[::-1])
#
# #create data
# data = np.load('/Users/Jason/Downloads/test.npy')
#
# di = griddata((x, y), data, (xi[None,:], yi[:,None]), method='cubic')
# di = np.ma.masked_invalid(di)
#
# # plot twoD_Gaussian data generated above
# # plt.figure()
# # plt.imshow(di)
# # plt.colorbar()
#
# # add some noise to the data and try to fit the data generated beforehand
# initial_guess = (3, 0, 0, 0.01, 0.01, 0, 10)
# bounds = ([0, x.min(), y.min(), 0, 0, -np.inf, -np.inf], np.inf)
#
# popt, pcov = optimize.curve_fit(twoD_Gaussian, (x, y), data, p0=initial_guess, bounds=bounds)
#
# data_fitted = twoD_Gaussian((x, y), *popt)
#
# embed()
#
# fig, ax = plt.subplots(1, 1)
# ax.hold(True)
# ax.imshow(data.reshape(xx.shape), cmap=plt.cm.jet, origin='bottom',
#     extent=(x.min(), x.max(), y.min(), y.max()))
# ax.contour(x, y, data_fitted.reshape(xx.shape), 8, colors='w')
# plt.show()


exe = JupiterRMS()
exe.run()
