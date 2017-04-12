from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from targetpipe.io.pixels import checm_pixel_pos, optical_foclen
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt


def plot_quick_camera(image, ax=None):
    ax = ax if ax is not None else plt.gca()

    pos = checm_pixel_pos * u.m
    foclen = optical_foclen * u.m
    geom = CameraGeometry.guess(*pos, foclen)
    camera = CameraDisplay(geom, ax=ax, image=image, cmap='viridis')
    return camera
