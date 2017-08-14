from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


class Scatter:
    def __init__(self, plot_name):
        self.plot_name = plot_name

        sns.set_style("white")
        sns.set_style("ticks")
        sns.set_context("paper", rc={"font.famly": "Helvetica",
                                     "font.size": 20,
                                     "axes.titlesize": 20,
                                     "axes.labelsize": 20
                                     })

        self.shape = 'square'

        if self.shape == 'wide':
            self.fig = plt.figure(figsize=(7, 5))
        else:
            self.fig = plt.figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.tick_params(labelsize=19)

    def add(self, x, y, y_err=None, label='', c=None):
        if not c:
            c = self.ax._get_lines.get_next_color()
        (_, caps, _) = self.ax.errorbar(x, y, yerr=y_err, fmt='o', mew=0.5,
                                        color=c, alpha=0.8, markersize=3,
                                        capsize=3, label=label)

        for cap in caps:
            cap.set_markeredgewidth(1)

    def create(self, x_label="", y_label="", title=""):
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.fig.suptitle(title)

    def add_xy_line(self):
        lims = [
            np.min([self.ax.get_xlim(), self.ax.get_ylim()]),
            np.max([self.ax.get_xlim(), self.ax.get_ylim()]),
        ]

        self.ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)
        self.ax.set_xlim(lims)
        self.ax.set_ylim(lims)

    def set_x_log(self):
        self.ax.set_xscale('log')
        formatter = FuncFormatter(lambda x, _: '{:g}'.format(x))
        self.ax.get_xaxis().set_major_formatter(formatter)

    def set_y_log(self):
        self.ax.set_yscale('log')
        formatter = FuncFormatter(lambda y, _: '{:g}'.format(y))
        self.ax.get_yaxis().set_major_formatter(formatter)

    def add_legend(self, loc=2):
        self.ax.legend(loc=loc)

    def save(self):
        self.fig.savefig(self.plot_name, bbox_inches='tight')


def main():
    laser_1825_file = np.load("pix1825_dr.npz")
    laser_1203_file = np.load("pix1203_dr.npz")
    led_1825_file = np.load("pix1825_dr_led.npz")
    led_1203_file = np.load("pix1203_dr_led.npz")

    # Plot Laser Dynamic Range
    p_laser_dr = Scatter("laser_dr.pdf")
    p_laser_dr.create("Illumination (p.e.)", "Charge (p.e.)", "")
    x = laser_1825_file['x']
    y = laser_1825_file['y']
    y_err = laser_1825_file['y_err']
    p_laser_dr.add(x, y, y_err, "Pixel 1825")
    x = laser_1203_file['x']
    y = laser_1203_file['y']
    y_err = laser_1203_file['y_err']
    p_laser_dr.add(x, y, y_err, "Pixel 1203")
    p_laser_dr.set_x_log()
    p_laser_dr.set_y_log()
    p_laser_dr.add_xy_line()
    p_laser_dr.add_legend()
    p_laser_dr.save()

    # Plot LED Dynamic Range
    p_led_dr = Scatter("led_dr.pdf")
    p_led_dr.create("LED", "Charge (p.e.)", "")
    x = led_1825_file['x']
    y = led_1825_file['y']
    y_err = led_1825_file['y_err']
    p_led_dr.add(x, y, y_err, "Pixel 1825")
    x = led_1203_file['x']
    y = led_1203_file['y']
    y_err = led_1203_file['y_err']
    p_led_dr.add(x, y, y_err, "Pixel 1203")
    p_led_dr.set_y_log()
    p_led_dr.add_legend()
    p_led_dr.save()


if __name__ == '__main__':
    main()
