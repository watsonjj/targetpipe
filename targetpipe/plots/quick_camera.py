from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from targetpipe.io.camera import Config
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt


def plot_quick_camera(image, ax=None):
    ax = ax if ax is not None else plt.gca()

    cameraconfig = Config()
    pos = cameraconfig.pixel_pos * u.m
    foclen = cameraconfig.optical_foclen * u.m
    geom = CameraGeometry.guess(*pos, foclen)
    camera = CameraDisplay(geom, ax=ax, image=image, cmap='viridis')
    return camera
