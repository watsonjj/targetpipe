from targetpipe.calib.camera.adc2pe import TargetioADC2PECalibrator
from targetpipe.io.pixels import Dead
import numpy as np
from matplotlib import pyplot as plt
from traitlets import Dict, List, Unicode
from ctapipe.core import Tool
from os.path import exists, join
from os import makedirs
import seaborn as sns
from pandas import DataFrame


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
        self.hv = None

        self.fig = None

        sns.set_style("whitegrid")

        self.adc2pe = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.a2p = TargetioADC2PECalibrator(**kwargs)
        self.dead = Dead()

        # hv_modules = [[819, 775, 784, 844, 772, 761, 785, 865, 863, 881, 715, 788, 725, 801, 828, 844, 793, 792, 835, 773, 774, 862, 803, 864, 788, 848, 766, 820, 773, 789, 845, 819],
        #               [914, 864, 885, 950, 869, 851, 887, 974, 977, 991, 812, 885, 819, 897, 932, 946, 892, 886, 943, 871, 866, 968, 902, 974, 883, 959, 864, 925, 873, 891, 952, 925],
        #               [1009, 954, 985, 1057, 966, 940, 989, 1084, 1092, 1101, 910, 981, 913, 992, 1035, 1048, 991, 979, 1052, 969, 957, 1075, 1001, 1085, 977, 1071, 963, 1030, 973, 993, 1059, 1032]]
        hv_modules = [[ 820, 771, 784, 848, 770, 759, 786, 870, 868, 887, 712, 788, 723, 802, 830, 847, 794, 793, 839, 773, 773, 867, 805, 869, 786, 853, 765, 822, 773, 782, 848, 821 ],
                      [ 916, 860, 885, 954, 866, 848, 887, 980, 982, 997, 809, 885, 816, 898, 934, 949, 893, 886, 947, 871, 864, 973, 904, 979, 882, 964, 864, 928, 873, 883, 955, 928 ],
                      [ 1011, 949, 986, 1061, 963, 937, 989, 1089, 1098, 1100, 907, 981, 909, 993, 1038, 1052, 992, 979, 1056, 969, 956, 1079, 1003, 1090, 977, 1075, 962, 1033, 973, 985, 1063, 1035 ]]
        shape = (3, 32, 64)
        hv_gm = (np.array(hv_modules)[..., None] *
                 np.ones(shape)).reshape((3, 2048))
        hv_modules = [[800]*2048,
                      [900]*2048,
                      [1000]*2048,
                      [1100]*2048]
        hv_ngm = np.array(hv_modules)
        self.hv = np.vstack((hv_ngm, hv_gm))

    def start(self):
        self.adc2pe = self.a2p.get_adc2pe_at_hv(self.hv, np.arange(2048)[None, :])
        self.spe = self.dead.mask2d(1/self.adc2pe)

        # Build Dataframe
        hv_df = np.array([[800]*2048,
                         [900]*2048,
                         [1000]*2048,
                         [1100]*2048,
                         [800]*2048,
                         [900]*2048,
                         [1000]*2048]).flatten()
        spe_df = self.spe.flatten()
        gm_df = ['Non-gain-matched']*2048*4 + ['Gain-matched']*2048*3
        d = dict(hv=hv_df, spe=spe_df, gm=gm_df)
        df = DataFrame(d)

        # Create Plot
        self.fig = plt.figure(figsize=(13, 6))
        ax = self.fig.add_subplot(1, 1, 1)
        sns.violinplot(ax=ax, data=df, x='hv', y='spe', hue='gm',
                       split=True, scale='count', inner='quartile')
        ax.set_xlabel('HV')
        ax.set_ylabel('SPE Value (ADC)')

    def finish(self):
        # Save figures
        output_dir = join(self.output_dir, "plot_adc2pe_vs_hv")
        if not exists(output_dir):
            self.log.info("Creating directory: {}".format(output_dir))
            makedirs(output_dir)

        numpy_path = join(output_dir, "spe_vs_hv.npz")
        fig_path = join(output_dir, "spe_vs_hv.pdf")

        x = np.array([790, 890, 990, 1090, 810, 910, 1010])
        sort = [0, 4, 1, 5, 2, 6, 3]
        x = x[sort]
        spe = self.spe[sort]

        np.savez(numpy_path,
                 charge=np.ma.filled(spe, 0),
                 charge_error=np.zeros(spe.shape),
                 rundesc=x)
        self.log.info("Numpy array saved to: {}".format(numpy_path))
        self.fig.savefig(fig_path)
        self.log.info("Figure saved to: {}".format(fig_path))

        adc2pe_800_path = join(output_dir, "adc2pe_800.npy")
        adc2pe_900_path = join(output_dir, "adc2pe_900.npy")
        adc2pe_1000_path = join(output_dir, "adc2pe_1000.npy")
        adc2pe_1100_path = join(output_dir, "adc2pe_1100.npy")
        adc2pe_800gm_path = join(output_dir, "adc2pe_800gm.npy")
        adc2pe_900gm_path = join(output_dir, "adc2pe_900gm.npy")
        adc2pe_1000gm_path = join(output_dir, "adc2pe_1000gm.npy")

        np.save(adc2pe_800_path, np.ma.filled(self.adc2pe[0], 0))
        self.log.info("ADC2PE array saved to: {}".format(adc2pe_800_path))
        np.save(adc2pe_900_path, np.ma.filled(self.adc2pe[1], 0))
        self.log.info("ADC2PE array saved to: {}".format(adc2pe_900_path))
        np.save(adc2pe_1000_path, np.ma.filled(self.adc2pe[2]))
        self.log.info("ADC2PE array saved to: {}".format(adc2pe_1000_path))
        np.save(adc2pe_1100_path, np.ma.filled(self.adc2pe[3]))
        self.log.info("ADC2PE array saved to: {}".format(adc2pe_1100_path))
        np.save(adc2pe_800gm_path, np.ma.filled(self.adc2pe[4]))
        self.log.info("ADC2PE array saved to: {}".format(adc2pe_800gm_path))
        np.save(adc2pe_900gm_path, np.ma.filled(self.adc2pe[5]))
        self.log.info("ADC2PE array saved to: {}".format(adc2pe_900gm_path))
        np.save(adc2pe_1000gm_path, np.ma.filled(self.adc2pe[6]))
        self.log.info("ADC2PE array saved to: {}".format(adc2pe_1000gm_path))


if __name__ == '__main__':
    exe = ADC2PEvsHVPlotter()
    exe.run()
