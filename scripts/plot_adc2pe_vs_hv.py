from targetpipe.calib.camera.adc2pe import TargetioADC2PECalibrator
from targetpipe.io.pixels import Dead
import numpy as np
from matplotlib import pyplot as plt
from traitlets import Dict, List, Unicode
from ctapipe.core import Tool
from os.path import dirname, exists, join
from os import makedirs


class ADC2PEvsHVPlotter(Tool):
    name = "ADC2PEvsHVPlotter"
    description = "For a given hv values, see the conversion from adc " \
                  "to pe for each pixel."

    output_dir = Unicode('', help='Directory to store output').tag(config=True)

    aliases = Dict(dict(adc2pe='TargetioADC2PECalibrator.adc2pe_path',
                        gm='TargetioADC2PECalibrator.gain_matching_path',
                        o='ADC2PEvsHVPlotter.output_dir'
                        ))

    classes = List([TargetioADC2PECalibrator,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.a2p = None
        self.dead = None
        self.spe = None
        self.fig = None

        hv_modules = [[819, 775, 784, 844, 772, 761, 785, 865, 863, 881, 715, 788, 725, 801, 828, 844, 793, 792, 835, 773, 774, 862, 803, 864, 788, 848, 766, 820, 773, 789, 845, 819],
                      [914, 864, 885, 950, 869, 851, 887, 974, 977, 991, 812, 885, 819, 897, 932, 946, 892, 886, 943, 871, 866, 968, 902, 974, 883, 959, 864, 925, 873, 891, 952, 925],
                      [1009, 954, 985, 1057, 966, 940, 989, 1084, 1092, 1101, 910, 981, 913, 992, 1035, 1048, 991, 979, 1052, 969, 957, 1075, 1001, 1085, 977, 1071, 963, 1030, 973, 993, 1059, 1032],
                      [0]*32]
        shape = (4, 32, 64)
        self.hv_gm = (np.array(hv_modules)[..., None] * np.ones(shape)).reshape((4, 2048))

        hv_modules = [[800]*2048,
                      [900]*2048,
                      [1000]*2048,
                      [1100]*2048]
        self.hv_ngm = np.array(hv_modules)

        self.x = [790, 810, 890, 910, 990, 1010, 1090]
        self.hv = np.empty((self.hv_gm.shape[0] + self.hv_ngm.shape[0], 2048))
        self.hv[0::2] = self.hv_ngm
        self.hv[1::2] = self.hv_gm
        self.hv = self.hv[:-1]

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.a2p = TargetioADC2PECalibrator(**kwargs)
        self.dead = Dead()

    def start(self):
        self.spe = 1/self.a2p.get_adc2pe_at_hv(self.hv, np.arange(2048)[None, :])
        self.spe = self.dead.mask2d(self.spe)

        self.fig = plt.figure(figsize=(13, 6))
        ax = self.fig.add_subplot(1, 1, 1)
        ax.plot(self.x, self.spe, 'x')
        ax.set_xlabel('HV (left=Before TM Gain Matching, '
                      'right=After TM Gain Matching)')
        ax.set_ylabel('SPE Value (ADC)')

    def finish(self):
        # Save figures
        output_dir = join(self.output_dir, "plot_adc2pe_vs_hv")
        if not exists(output_dir):
            self.log.info("Creating directory: {}".format(output_dir))
            makedirs(output_dir)

        numpy_path = join(output_dir, "spe_vs_hv.npz")
        fig_path = join(output_dir, "spe_vs_hv.pdf")

        np.savez(numpy_path,
                 charge=np.ma.filled(self.spe, 0),
                 charge_error=np.zeros(self.spe.shape),
                 rundesc=self.x)
        self.log.info("Numpy array saved to: {}".format(numpy_path))
        self.fig.savefig(fig_path)
        self.log.info("Figure saved to: {}".format(fig_path))


if __name__ == '__main__':
    exe = ADC2PEvsHVPlotter()
    exe.run()