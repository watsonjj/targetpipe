import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
import os

lib = np.ctypeslib.load_library("correlate_c", os.path.dirname(__file__))
cross = lib.cross
cross.restype = None
cross.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_size_t, ctypes.c_size_t,
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_size_t,
    ctypes.c_int
]


def cross_correlate(waveforms, reference):
    cleaned = np.zeros(waveforms.shape, dtype=np.float32)
    cen = 0
    cross(waveforms.astype(np.float32), *waveforms.shape,
          cleaned, reference.astype(np.float32), reference.size, cen)
    return cleaned
