from tqdm import tqdm, trange
from traitlets import Dict, List
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from os.path import exists, join
from os import makedirs

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import AverageWfPeakIntegrator
from ctapipe.image.waveform_cleaning import CHECMWaveformCleanerAverage
from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.calib.camera.r1 import TargetioR1Calibrator
from targetpipe.fitting.chec import CHECBrightFitter
from targetpipe.io.pixels import Dead


class SpreadComparer(Tool):
    name = "SpreadComparer"
    description = "Compare the spread between different files"

    aliases = Dict(dict())
    classes = List([TargetioFileReader,
                    TargetioR1Calibrator,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dl0 = None
        self.dl1 = None
        self.fitter = None
        self.dead = None

        self.path_dict = dict()
        self.reader_dict = dict()
        self.r1_dict = dict()

        self.n_pixels = None
        self.n_samples = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.path_dict['800'] = '/Volumes/gct-jason/data/170310/hv/Run00904_r1_adc.tio'
        self.path_dict['900'] = '/Volumes/gct-jason/data/170310/hv/Run00914_r1_adc.tio'
        self.path_dict['1000'] = '/Volumes/gct-jason/data/170310/hv/Run00924_r1_adc.tio'
        self.path_dict['1100'] = '/Volumes/gct-jason/data/170310/hv/Run00934_r1_adc.tio'
        self.path_dict['800gm'] = '/Volumes/gct-jason/data/170319/gainmatching/gainmatched/Run03983_r1_adc.tio'
        self.path_dict['900gm'] = '/Volumes/gct-jason/data/170319/gainmatching/gainmatched/Run03984_r1_adc.tio'
        self.path_dict['1000gm'] = '/Volumes/gct-jason/data/170320/linearity/Run04174_r1_adc.tio'#'/Volumes/gct-jason/data/170319/gainmatching/gainmatched/Run03985_r1_adc.tio'

        # self.path_dict['800'] = '/Volumes/gct-jason/data/170310/hv/Run00904_r1.tio'
        # self.path_dict['900'] = '/Volumes/gct-jason/data/170310/hv/Run00914_r1.tio'
        # self.path_dict['1000'] = '/Volumes/gct-jason/data/170310/hv/Run00924_r1.tio'
        # self.path_dict['1100'] = '/Volumes/gct-jason/data/170310/hv/Run00934_r1.tio'
        # self.path_dict['800gm'] = '/Volumes/gct-jason/data/170319/gainmatching/gainmatched/Run03983_r1.tio'
        # self.path_dict['900gm'] = '/Volumes/gct-jason/data/170319/gainmatching/gainmatched/Run03984_r1.tio'
        # self.path_dict['1000gm'] = '/Volumes/gct-jason/data/170320/linearity/Run04174_r1.tio'

        ped = '/Volumes/gct-jason/data/170310/pedestal/Run00843_ped.tcal'
        pedgm = '/Volumes/gct-jason/data/170319/gainmatching/pedestal/Run03932_ped.tcal'
        tf = '/Volumes/gct-jason/data/170310/tf/Run00844-00893_tf.tcal'

        for key, val in self.path_dict.items():
            # p = ped
            # if 'gm' in key:
            #     p = pedgm
            # adc2pe = '/Users/Jason/Software/CHECAnalysis/targetpipe/adc2pe/adc2pe_{}.tcal'.format(key)
            # self.r1_dict[key] = TargetioR1Calibrator(pedestal_path=p,
            #                                          tf_path=tf,
            #                                          adc2pe_path='',
            #                                          **kwargs)
            self.reader_dict[key] = TargetioFileReader(input_path=val,
                                                       **kwargs)

        cleaner = CHECMWaveformCleanerAverage(**kwargs)
        extractor = AverageWfPeakIntegrator(**kwargs)
        self.dl0 = CameraDL0Reducer(**kwargs)
        self.dl1 = CameraDL1Calibrator(extractor=extractor,
                                       cleaner=cleaner,
                                       **kwargs)
        self.fitter = CHECBrightFitter(**kwargs)
        self.dead = Dead()

        first_event = list(self.reader_dict.values())[0].get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        r1 = first_event.r1.tel[telid].pe_samples[0]
        self.n_pixels, self.n_samples = r1.shape

    def start(self):
        df_list = []
        tm9_1100 = None
        tm9_1000gm = None
        tm0_1100 = None
        tm0_1000gm = None


        desc1 = 'Looping through files'
        for key, reader in tqdm(self.reader_dict.items(), desc=desc1):
            hv = int(key.replace("gm", ""))
            gm = 'gm' in key
            gm_t = 'Gain-matched' if 'gm' in key else 'Non-gain-matched'

            source = reader.read()
            n_events = reader.num_events

            dl1 = np.zeros((n_events, self.n_pixels))

            desc2 = "Extracting Charge"
            for event in tqdm(source, desc=desc2, total=n_events):
                ev = event.count
                # self.r1_dict[key].calibrate(event)
                self.dl0.reduce(event)
                self.dl1.calibrate(event)
                dl1[ev] = event.dl1.tel[0].image[0]

            dl1_modules = dl1.reshape((n_events, 32, 64))
            if key == '1100':
                tm9_1100 = dl1_modules[:, 9]
                tm0_1100 = dl1_modules[:, 0]
            elif key == '1000gm':
                tm9_1000gm = dl1_modules[:, 9]
                tm0_1000gm = dl1_modules[:, 0]

            desc3 = "Fitting Pixels"
            for pix in trange(self.n_pixels, desc=desc3):
                pixel_area = dl1[:, pix]
                if pix in self.dead.dead_pixels:
                    continue
                # if not self.fitter.apply(pixel_area):
                #     self.log.warning("File {} Pixel {} could not be fitted"
                #                      .format(key, pix))
                #     continue
                charge = np.median(pixel_area)#self.fitter.coeff['mean']
                df_list.append(dict(key=key, hv=hv, gm=gm, gm_t=gm_t,
                                    pixel=pix, tm=pix//64, charge=charge))

        df = pd.DataFrame(df_list)

        # where_gm = df['gm'] == True
        # df.loc[where_gm, 'charge'] /= tm9_1000gm.mean()
        # df.loc[~where_gm, 'charge'] /= tm9_1100.mean()

        df = df.sort_values(by=['gm', 'hv'], ascending=[True, True])

        # Save figures
        output_dir = "/Volumes/gct-jason/plots/compare_gain_matching_spread"
        if not exists(output_dir):
            self.log.info("Creating directory: {}".format(output_dir))
            makedirs(output_dir)

        # Create Violin Plot
        fig_v = plt.figure(figsize=(13, 6))
        ax = fig_v.add_subplot(1, 1, 1)
        sns.violinplot(ax=ax, data=df, x='hv', y='charge', hue='gm_t',
                       split=True, scale='count', inner='quartile')
        ax.set_xlabel('HV')
        ax.set_ylabel('Charge (ADC after Pedestal & TF)')

        # Create TM9 1100V comparison plot
        fig_TM9_1100V = plt.figure(figsize=(13, 6))
        ax = fig_TM9_1100V.add_subplot(1, 1, 1)
        ax.hist(tm9_1100.ravel()/tm9_1100.mean(), bins=100, range=[0,2], alpha=0.4, label='1100')
        ax.hist(tm9_1000gm.ravel()/tm9_1000gm.mean(), bins=100, range=[0,2], alpha=0.4, label='1000gm')
        ax.set_xlabel('Charge (ADC after Pedestal & TF)')
        ax.set_ylabel('N')
        ax.set_title('TM9 @ 1100V')
        ax.legend(loc='upper right')

        # Create TM0 1100V comparison plot
        fig_TM0_1100V = plt.figure(figsize=(13, 6))
        ax = fig_TM0_1100V.add_subplot(1, 1, 1)
        ax.hist(tm0_1100.ravel()/tm9_1100.mean(), bins=100, range=[0,3], alpha=0.4, label='1100')
        ax.hist(tm0_1000gm.ravel()/tm9_1000gm.mean(), bins=100, range=[0,3], alpha=0.4, label='1000gm')
        ax.set_xlabel('Charge (ADC after Pedestal & TF)')
        ax.set_ylabel('N')
        ax.set_title('TM0 @ 1100V')
        ax.legend(loc='upper right')

        # Create module mean plot
        fig_tm_mean = plt.figure(figsize=(13, 6))
        ax = fig_tm_mean.add_subplot(1, 1, 1)
        for key in self.reader_dict.keys():
            means = np.zeros(32)
            for tm in range(32):
                means[tm] = df[df['key'] == key][df['tm'] == tm]['charge'].mean()
            if '800' in key:
                c = 'red'
            elif '900' in key:
                c = 'blue'
            elif '1000' in key:
                c = 'green'
            else:
                c = 'black'
            m = 'd' if 'gm' in key else 'o'
            ax.plot(np.arange(32), means, linestyle='-', marker=m, color=c, label=key)
        ax.set_xlabel('TM')
        ax.set_ylabel('Charge (ADC after Pedestal & TF)')
        ax.set_title('TM means')
        ax.legend(loc='upper right')

        fig_path = join(output_dir, "compare_gain_matching_spread.pdf")
        fig_v.savefig(fig_path)
        self.log.info("Figure saved to: {}".format(fig_path))
        fig_path = join(output_dir, "TM9_1100V_comparison.pdf")
        fig_TM9_1100V.savefig(fig_path)
        self.log.info("Figure saved to: {}".format(fig_path))
        fig_path = join(output_dir, "TM0_1100V_comparison.pdf")
        fig_TM0_1100V.savefig(fig_path)
        self.log.info("Figure saved to: {}".format(fig_path))
        fig_path = join(output_dir, "TM_means.pdf")
        fig_tm_mean.savefig(fig_path)
        self.log.info("Figure saved to: {}".format(fig_path))

    def finish(self):
        pass


if __name__ == '__main__':
    exe = SpreadComparer()
    exe.run()
