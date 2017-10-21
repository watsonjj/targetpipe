from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
from IPython import embed

class Geometry:
    def __init__(self):
        self.fig = plt.figure(figsize=(13, 13))
        self.ax = self.fig.add_subplot(111)

        self.camera_curve_center_x = 1
        self.camera_curve_center_y = 0
        self.camera_curve_radius = 1
        self.pixel_pos = np.load("/Users/Jason/Software/CHECAnalysis/targetpipe/targetpipe/io/checm_pixel_pos.npy")
        self.pix_size = 0.006125
        self.fiducial_radius = 0.2
        self.fiducial_center = self.camera_curve_center_x + self.camera_curve_radius * np.cos(np.pi)
        self.lightsource_distance = 1

        theta_f = np.pi / 2 - np.arccos(self.fiducial_radius / self.lightsource_distance)
        theta_ls = theta_f#self.lightsource_angle * np.pi / 180
        theta_p = np.arctan(self.pix_size / (2 * self.lightsource_distance))
        self.lightsource_angle = theta_f * 180 / np.pi
        self.lightsource_sa = 2 * np.pi * (1 - np.cos(theta_ls))
        self.fiducial_sa = 2 * np.pi * (1 - np.cos(theta_f))
        self.fiducial_percent = self.fiducial_sa / self.lightsource_sa
        if self.fiducial_percent > 1: self.fiducial_percent = 1
        self.pix_sa = 2 * np.pi * (1 - np.cos(theta_p))
        self.pix_percent = self.pix_sa / self.lightsource_sa
        if self.pix_percent > 1: self.pix_percent = 1

        photons = self.set_illumination(1000)
        print(photons)
        print(self.lightsource_angle)

    def set_illumination(self, lambda_):
        self.lamda = lambda_
        self.pde = 0.3936
        self.photons_pixel = self.lamda / self.pde
        self.photons_ls = self.photons_pixel / self.pix_percent
        self.photons_fiducial = self.photons_ls * self.fiducial_percent
        return self.photons_ls

    def plot_camera_circle(self):
        x = self.camera_curve_center_x
        y = self.camera_curve_center_y
        r = self.camera_curve_radius
        self.ax.add_artist(plt.Circle((x, y), r, color='b', fill=False))

    def plot_pixels(self):
        pix_x = np.unique(np.reshape(self.pixel_pos[0], (32, 64))[10:16])
        pix_cy = np.reshape(pix_x, (6, 8))
        module_centers = pix_cy.mean(1)

        x0 = self.camera_curve_center_x
        y0 = self.camera_curve_center_y
        r = self.camera_curve_radius

        mod_y = module_centers
        theta = np.arcsin((mod_y - y0)/r) + np.pi
        x = x0 + r * np.cos(theta)
        mr = (mod_y - y0) / (x - x0)
        mt = -1 / mr
        pix_cx = ((pix_cy - mod_y[:, None]) / mt[:, None]) + x[:, None]
        pix_x1 = pix_cx - (self.pix_size / 2) / np.sqrt(1 + mt[:, None] ** 2)
        pix_x2 = pix_cx + (self.pix_size / 2) / np.sqrt(1 + mt[:, None] ** 2)
        pix_y1 = (pix_x1 - x[:, None]) * mt[:, None] + mod_y[:, None]
        pix_y2 = (pix_x2 - x[:, None]) * mt[:, None] + mod_y[:, None]
        pix_x = np.vstack([pix_x1.ravel(), pix_x2.ravel()])
        pix_y = np.vstack([pix_y1.ravel(), pix_y2.ravel()])

        # from IPython import embed
        # embed()

        self.ax.plot(pix_x, pix_y, color='r')

    def plot_focal_plane(self):
        self.ax.axvline(self.fiducial_center)

    def plot_fiducial_sphere(self):
        x0 = self.fiducial_center
        y0 = self.camera_curve_center_y
        r = self.fiducial_radius
        self.ax.add_artist(plt.Circle((x0, y0), r, color='b', fill=False))

    def plot_lightsource(self):
        x = self.fiducial_center - self.lightsource_distance
        y = self.camera_curve_center_y
        self.ax.plot(x, y, 'x')
        angle = self.lightsource_angle
        wedge = patches.Wedge((x, y), 2*self.lightsource_distance, -angle, angle, alpha=0.5)
        self.ax.add_patch(wedge)

    def plot_values(self):
        text = "Lightsource = {} photons \n" \
               "Fiducial = {} photons \n" \
               "Pixel = {} photons \n" \
               "Pixel = {} p.e.".format(self.photons_ls, self.photons_fiducial,
                                        self.photons_pixel, self.lamda)
        self.ax.text(0.5, 0.5, text, transform=self.ax.transAxes)

    def plot(self):
        self.plot_camera_circle()
        self.plot_pixels()
        self.plot_focal_plane()
        self.plot_fiducial_sphere()
        self.plot_lightsource()
        self.plot_values()
        # self.ax.set_xlim([-1, 0.05])
        # self.ax.set_ylim([-0.4, 0.4])
        plt.show()


def main():
    g = Geometry()
    g.plot()


if __name__ == '__main__':
    main()