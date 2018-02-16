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
from target_calib import CfMaker


class ADC2PEvsHVPlotter(Tool):
    name = "ADC2PEvsHVPlotter"
    description = "For a given hv values, get the conversion from adc " \
                  "to pe for each pixel."

    output_dir = Unicode('', help='Directory to store output').tag(config=True)

    aliases = Dict(dict(spe='TargetioADC2PECalibrator.spe_path',
                        gm='TargetioADC2PECalibrator.gain_matching_path',
                        o='ADC2PEvsHVPlotter.output_dir'
                        ))

    classes = List([TargetioADC2PECalibrator,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.a2p = None
        self.dead = None

        sns.set_style("whitegrid")

        self.cfmaker = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.a2p = TargetioADC2PECalibrator(**kwargs)
        self.dead = Dead()

        self.cfmaker = CfMaker(32)

        # Save figures
        self.output_dir = join(self.output_dir, "plot_adc2pe_vs_hv")
        if not exists(self.output_dir):
            self.log.info("Creating directory: {}".format(self.output_dir))
            makedirs(self.output_dir)

    def start(self):
        hv_dict = dict()
        hv_dict['800'] = [800]*2048
        hv_dict['900'] = [900]*2048
        hv_dict['1000'] = [1000]*2048
        hv_dict['1100'] = [1100]*2048
        hv_dict['800gm'] = [self.a2p.gm800[i//64] for i in range(2048)]
        hv_dict['900gm'] = [self.a2p.gm900[i//64] for i in range(2048)]
        hv_dict['1000gm'] = [self.a2p.gm1000[i//64] for i in range(2048)]
        hv_dict['800gm_c1'] = [self.a2p.gm800_c1[i//64] for i in range(2048)]

        df_list = []

        for key, hv in hv_dict.items():
            hv_group = int(key.replace("gm", "").replace("_c1", ""))
            gm = 'gm' in key
            gm_t = 'Gain-matched' if 'gm' in key else 'Non-gain-matched'
            adc2pe = self.a2p.get_adc2pe_at_hv(hv, np.arange(2048))
            adc2pe = self.dead.mask1d(adc2pe)
            spe = 1/adc2pe
            self.cfmaker.SetAll(np.ma.filled(adc2pe, 0).astype(np.float32))
            path = join(self.output_dir, "adc2pe_{}.tcal".format(key))
            self.cfmaker.Save(path, False)
            self.log.info("ADC2PE tcal created: {}".format(path))
            self.cfmaker.Clear()

            for pix in range(2048):
                if pix in self.dead.dead_pixels:
                    continue
                df_list.append(dict(pixel=pix, key=key, hv_group=hv_group,
                                    gm=gm, gm_t=gm_t, hv=hv[pix],
                                    adc2pe=adc2pe[pix], spe=spe[pix]))

        df = DataFrame(df_list)
        df = df.loc[df['key'] != '800gm_c1']

        # Create Plot
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(1, 1, 1)
        sns.violinplot(ax=ax, data=df, x='hv_group', y='spe', hue='gm_t',
                       split=True, scale='count', inner='quartile')
        ax.set_xlabel('HV')
        ax.set_ylabel('SPE Value (ADC)')
        fig_path = join(self.output_dir, "spe_vs_hv.pdf")
        fig.savefig(fig_path)

    def finish(self):
        pass


if __name__ == '__main__':
    exe = ADC2PEvsHVPlotter()
    exe.run()
