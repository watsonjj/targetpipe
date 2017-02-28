"""Handles the conversion between module,asic,channel and pixel_id. Also
handles the hard coding of pixel positions for the GCT camera.
"""

import numpy as np
from astropy import log
from os.path import dirname, realpath, join
from ctapipe.io.camera import get_min_pixel_seperation

log.info("Initialising pixel geometry and ids")

path = join(dirname(realpath(__file__)), 'checm_pixel_id.npy')
checm_pixel_id = np.load(path)
path = join(dirname(realpath(__file__)), 'checm_pixel_pos.npy')
checm_pixel_pos = np.load(path)

optical_foclen = 2.283


def pixel_sizes(x_pix, y_pix):
    """

    Parameters
    ----------
    x_pix
    y_pix

    Returns
    -------
    sizes : ndarray

    """
    dx = x_pix - x_pix[0]
    dy = y_pix - y_pix[0]
    pixsep = np.sqrt(dx ** 2 + dy ** 2)
    sizes = np.ones_like(x_pix) * np.min(pixsep[1:])
    return sizes


def get_pixel_2d(x_pix, y_pix, values=None):
    n_pix = x_pix.size
    if values is None:
        # By default, fill with pixel id
        values = np.arange(n_pix)

    gx = np.histogram2d(x_pix, y_pix, weights=x_pix, bins=[53, 53])[0]
    gy = np.histogram2d(x_pix, y_pix, weights=y_pix, bins=[53, 53])[0]
    xc = gx[:, 10][gx[:, 10].nonzero()]
    yc = gy[10, :][gy[10, :].nonzero()]

    dist = get_min_pixel_seperation(xc, yc)
    edges_x = np.zeros(xc.size + 1)
    edges_x[0:xc.size] = xc - dist / 2
    edges_x[-1] = xc[-1] + dist / 2
    edges_y = np.zeros(yc.size + 1)
    edges_y[0:yc.size] = yc - dist / 2
    edges_y[-1] = yc[-1] + dist / 2

    camera = np.histogram2d(-y_pix, x_pix, bins=[edges_x, edges_y],
                            weights=values+1)[0]
    camera[camera == 0] = np.nan
    camera -= 1
    return camera


def get_neighbours_2d(x_pix, y_pix):
    _2d = get_pixel_2d(x_pix, y_pix)
    pad = np.pad(_2d, 1, 'constant', constant_values=np.nan)
    neighbours = []
    for pix in range(2048):
        i, j = np.where(_2d == pix)
        i = i[0] + 1
        j = j[0] + 1
        d = 1
        nei = pad[i - d:i + d + 1, j - d:j + d + 1]
        neighbours.append(nei)
    return neighbours


# def invert_2d_array(array):
#     y, x = np.indices(array.shape)
#     positions = np.array(list(zip(y.ravel(), x.ravel())))
#     d = dict(zip(array.ravel(), positions))
#     l = list(d.values())
#     n = np.concatenate(l).reshape((2048, 2))
#     return n
#
#
# def invert_3d_array(array):
#     z, y, x = np.indices(array.shape)
#     positions = np.array(list(zip(z.ravel(), y.ravel(), x.ravel())))
#     d = dict(zip(array.ravel(), positions))
#     l = list(d.values())
#     n = np.concatenate(l).reshape((2048, 3))
#     return n
#
#
# class Pixels:
#     def __init__(self):
#         self.__module_asic_channel_to_pix = None
#         self.__pix_to_module_asic_channel = None
#         self.__tm_tmpix_to_pix = None
#         self.__pix_to_tm_tmpix = None
#         self.__pix_to_tm = None
#         self.__pix_to_tmpix = None
#         self.__pixel_coordinates = None
#         self.__pixel_sizes = None
#
#         self.n_a = 4
#         self.n_c = 16
#         self.n_tmpix = self.n_a * self.n_c
#         self.n_pix = pixel_id.size
#         self.n_m = self.n_pix / self.n_tmpix
#
#     @property
#     def module_asic_channel_to_pix(self):
#         if self.__module_asic_channel_to_pix is None:
#             array = np.array(pixel_id).reshape(self.n_m, self.n_a, self.n_c)
#             self.__module_asic_channel_to_pix = array
#         return self.__module_asic_channel_to_pix
#
#     @property
#     def pix_to_module_asic_channel(self):
#         if self.__pix_to_module_asic_channel is None:
#             array = invert_3d_array(self.module_asic_channel_to_pix)
#             self.__pix_to_module_asic_channel = array
#         return self.__pix_to_module_asic_channel
#
#     @property
#     def tm_tmpix_to_pix(self):
#         if self.__tm_tmpix_to_pix is None:
#             array = np.array(pixel_id).reshape(self.n_m, self.n_tmpix)
#             self.__tm_tmpix_to_pix = array
#         return self.__tm_tmpix_to_pix
#
#     @property
#     def pix_to_tm_tmpix(self):
#         if self.__pix_to_tm_tmpix is None:
#             array = invert_2d_array(self.tm_tmpix_to_pix)
#             self.__pix_to_tm_tmpix = array
#         return self.__pix_to_tm_tmpix
#
#     @property
#     def pixel_coordinates(self):
#         if self.__pixel_coordinates is None:
#             array = np.rollaxis(np.array(pixel_pos), 1, 0)
#             self.__pixel_coordinates = array
#         return self.__pixel_coordinates
#
#     @property
#     def pixel_sizes(self):
#         if self.__pixel_sizes is None:
#             x_pix, y_pix = self.pixel_coordinates
#             dx = x_pix - x_pix[0]
#             dy = y_pix - y_pix[0]
#             pixsep = np.sqrt(dx ** 2 + dy ** 2)
#             self.__pixel_sizes = np.ones_like(x_pix) * np.min(pixsep[1:])
#         return self.__pixel_sizes
#
#     def convert_module_asic_channel_to_pix(self, module, asic, channel):
#         return self.module_asic_channel_to_pix[module, asic, channel]
#
#     def convert_pix_to_module_asic_channel(self, pix):
#         return self.pix_to_module_asic_channel[pix]
#
#     def convert_tm_tmpix_to_pix(self, tm, tmpix):
#         return self.tm_tmpix_to_pix[tm, tmpix]
#
#     def convert_pix_to_tm_tmpix(self, pix):
#         return self.pix_to_tm_tmpix[pix]
#
#     def get_pixel_pos(self, pix):
#         return self.pixel_coordinates[pix]
