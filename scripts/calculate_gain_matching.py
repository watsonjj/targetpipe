import iminuit

from targetpipe.io.pixels import Dead
import numpy as np
from matplotlib import pyplot as plt
from traitlets import Dict, List, Unicode
from ctapipe.core import Tool
from os.path import exists, join
from os import makedirs
import seaborn as sns
from pandas import DataFrame
from functools import partial


class ADC2PEvsHVPlotter(Tool):
    name = "ADC2PEvsHVPlotter"
    description = "For a given hv values, see the conversion from adc " \
                  "to pe for each pixel."

    gain_matching_path = Unicode('', help='Path to the numpy gain matching '
                                          'file').tag(config=True)
    output_dir = Unicode('', help='Directory to store output').tag(config=True)

    aliases = Dict(dict(gm='ADC2PEvsHVPlotter.gain_matching_path',
                        o='ADC2PEvsHVPlotter.output_dir'
                        ))

    classes = List([
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.c = None
        self.alpha = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        gain_matching = np.load(self.gain_matching_path)
        self.c = gain_matching['C_tm']
        self.alpha = gain_matching['alpha_tm']

    def start(self):
        print("JASON METHOD")
        hv = 800
        gm_800 = self.gain_match(self.c, self.alpha, hv).astype(np.int)
        print('[', ', '.join(map(str, gm_800)), ']')

        hv = 900
        gm_900 = self.gain_match(self.c, self.alpha, hv).astype(np.int)
        print('[', ', '.join(map(str, gm_900)), ']')

        hv = 1000
        gm_1000 = self.gain_match(self.c, self.alpha, hv).astype(np.int)
        print('[', ', '.join(map(str, gm_1000)), ']')

        hv = 1100
        gm_1100 = self.gain_match(self.c, self.alpha, hv).astype(np.int)
        print('[', ', '.join(map(str, gm_1100)), ']')

        print("JUSTUS METHOD")
        hv = 800
        gm_800 = self.gain_match_justus(self.c, self.alpha, hv).astype(np.int)
        print('[', ', '.join(map(str, gm_800)), ']')

        hv = 900
        gm_900 = self.gain_match_justus(self.c, self.alpha, hv).astype(np.int)
        print('[', ', '.join(map(str, gm_900)), ']')

        hv = 1000
        gm_1000 = self.gain_match_justus(self.c, self.alpha, hv).astype(np.int)
        print('[', ', '.join(map(str, gm_1000)), ']')

        # filename = "Measurement_2/hvSetting_%i.cfg" % st
        # f = open(filename, 'w')
        # for s in slots:
        #     f.write("M:%i/F|HV=%i\n" % (s, hv_setting[st][s]))
        # f.close()

    def finish(self):
        pass

    @staticmethod
    def x_function(c, alpha, x):
        return np.mean(c * np.power(x, alpha))

    @staticmethod
    def y_function_arr(c, alpha, y):
        x_val = np.round(np.power(y / c, 1 / alpha))
        x_val[x_val > 1100] = 1100
        x_val[x_val < 0] = 0
        return x_val

    @staticmethod
    def y_function(c, alpha, y):
        return np.mean(ADC2PEvsHVPlotter.y_function_arr(c, alpha, y))

    def gain_match(self, c, alpha, goal_x):
        f_x = partial(self.x_function, c=c, alpha=alpha)

        def m(try_y):
            return np.abs(self.y_function(c, alpha, try_y) - goal_x)

        p0 = dict(try_y=f_x(x=goal_x))
        limits = dict(limit_try_y=(f_x(x=0), f_x(x=1500)))
        m0 = iminuit.Minuit(m, **p0, **limits,
                            print_level=-1, pedantic=False, throw_nan=True)
        m0.migrad()
        y = m0.args[0]
        best_x = self.y_function(c, alpha, y)
        print("Goal: {}, Best: {}".format(goal_x, best_x))
        return self.y_function_arr(c, alpha, y)

    def gain_match_justus(self, c, alpha, goal_x):
        mean_G = np.mean(self.x_function(c, alpha, goal_x))
        mean_hv = self.y_function(c, alpha, mean_G)

        print("Goal: {}, Best: {}".format(goal_x, mean_hv))
        return self.y_function_arr(c, alpha, mean_G)

if __name__ == '__main__':
    exe = ADC2PEvsHVPlotter()
    exe.run()
