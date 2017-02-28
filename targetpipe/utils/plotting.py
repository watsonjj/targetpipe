from matplotlib.cm import viridis
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
import os

lib = np.ctypeslib.load_library("rgbtohex", os.path.dirname(__file__))
rgbtohex = lib.rgbtohex
rgbtohex.restype = None
rgbtohex.argtypes = [ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
                     ctypes.c_size_t,
                     ndpointer(ctypes.c_char, flags="C_CONTIGUOUS")]


def intensity_to_rgb(array, minval=None, maxval=None):
    """
    Converts the values of an array to rgb representing a color for a z axis

    Parameters
    ----------
    array : ndarray or list
        1D numpy array or list containing intensity values for a z axis
    minval: int
        minimum value of the image
    maxval: int
        maximum value of the image

    Returns
    -------
    rgb : ndarray
        rgb tuple representing the intensity as a color

    """
    if minval is None:
        minval = array.min()
    if maxval is None:
        maxval = array.max()
    if maxval == minval:
        minval -= 1
        maxval += 1
    scaled = (array - minval) / (maxval - minval)

    rgb = (255 * viridis(scaled)).astype(np.uint8)
    return rgb


def intensity_to_hex(array, minval=None, maxval=None):
    """
    Converts the values of an array to hex representing a color for a z axis

    Parameters
    ----------
    array : ndarray or list
        1D numpy array or list containing intensity values for a z axis
    minval: int
        minimum value of the image
    maxval: int
        maximum value of the image

    Returns
    -------
    hex_ : ndarray
        hex strings representing the intensity as a color

    """
    # hex_ = ["#%02x%02x%02x" % (int(r), int(g), int(b))
    #         for r, g, b, _ in 255 * mpl.cm.viridis(scaled)]
    array_size = array.size
    hex_ = np.empty((array_size, 8), dtype='S1')
    rgbtohex(intensity_to_rgb(array, minval, maxval), array_size, hex_)
    return hex_.view('S8').astype('U8')[:, 0]
