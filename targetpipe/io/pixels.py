"""
Handles everything to do with pixels and geometries.
"""

import numpy as np
from ctapipe.instrument.camera import _get_min_pixel_seperation
from ctapipe.instrument import CameraGeometry
from astropy import units as u
from targetpipe.io.camera import Config


def get_geometry():
    cameraconfig = Config()
    pixel_pos = cameraconfig.pixel_pos
    optical_foclen = cameraconfig.optical_foclen
    return CameraGeometry.guess(*pixel_pos * u.m, optical_foclen * u.m)


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
    i = np.bincount(gx.nonzero()[0]).argmax()
    j = np.bincount(gy.nonzero()[0]).argmax()
    xc = gx[:, i][gx[:, i].nonzero()]
    yc = gy[j, :][gy[j, :].nonzero()]

    dist = _get_min_pixel_seperation(xc, yc)
    edges_x = np.zeros(xc.size + 1)
    edges_x[0:xc.size] = xc - dist / 2
    edges_x[-1] = xc[-1] + dist / 2
    edges_y = np.zeros(yc.size + 1)
    edges_y[0:yc.size] = yc - dist / 2
    edges_y[-1] = yc[-1] + dist / 2

    camera = np.histogram2d(-y_pix, x_pix, bins=[-edges_y[::-1], edges_x],
                            weights=values+1)[0]
    camera[camera == 0] = np.nan
    camera -= 1
    return camera


def get_neighbours_2d(x_pix, y_pix):
    _2d = get_pixel_2d(x_pix, y_pix)
    pad = np.pad(_2d, 1, 'constant', constant_values=np.nan)
    neighbours = []
    for pix in range(x_pix.size):
        i, j = np.where(_2d == pix)
        i = i[0] + 1
        j = j[0] + 1
        d = 1
        nei = pad[i - d:i + d + 1, j - d:j + d + 1]
        neighbours.append(nei)
    return neighbours


class Dead:
    def __init__(self):
        cameraconfig = Config()
        self.dead_pixels = cameraconfig.dead_pixels
        self.n_pix = cameraconfig.n_pix

    def get_pixel_mask(self):
        mask = np.zeros(self.n_pix, dtype=np.bool)
        mask[self.dead_pixels] = True
        return mask

    def mask1d(self, array):
        if not array.shape[0] == self.n_pix:
            print("[ERROR] array does not contain {} pixels, "
                  "cannot mask dead".format(self.n_pix))
            return array

        mask = np.zeros(array.shape, dtype=np.bool)
        mask[self.dead_pixels] = True

        masked = array
        if not np.ma.isMaskedArray(masked):
            masked = np.ma.array(masked)
        masked.mask = np.ma.mask_or(masked.mask, mask)
        return masked

    def mask2d(self, array):
        if not array.shape[1] == self.n_pix:
            print("[ERROR] array does not contain {} pixels, "
                  "cannot mask dead".format(self.n_pix))
            return array

        mask = np.zeros(array.shape, dtype=np.bool)
        mask[:, self.dead_pixels] = True

        masked = array
        if not np.ma.isMaskedArray(masked):
            masked = np.ma.array(masked)
        masked.mask = np.ma.mask_or(masked.mask, mask)
        return masked
