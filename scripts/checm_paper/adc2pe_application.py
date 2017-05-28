from tqdm import tqdm, trange
from traitlets import Dict, List
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import seaborn as sns

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import AverageWfPeakIntegrator
from ctapipe.image.waveform_cleaning import CHECMWaveformCleanerAverage
from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.calib.camera.r1 import TargetioR1Calibrator
from targetpipe.fitting.chec import CHECBrightFitter
from targetpipe.io.pixels import Dead
from targetpipe.plots.official import OfficialPlotter


class ADC2PEPlotter(OfficialPlotter):
    name = 'ADC2PEPlotter'

    def __init__(self, config, tool, **kwargs):
        """
        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, tool=tool, **kwargs)

    def create(self, df):
        sns.violinplot(ax=self.ax, data=df, x='hv', y='charge', hue='cal_t',
                       split=True, scale='count', inner='quartile',
                       legend=False)
        self.ax.set_title("Distribution Before and After ADC2PE Correction")
        self.ax.set_xlabel('HV')
        self.ax.set_ylabel('Charge (p.e.)')
        self.ax.legend(loc="upper right")

        major_locator = MultipleLocator(50)
        major_formatter = FormatStrFormatter('%d')
        minor_locator = MultipleLocator(10)
        self.ax.yaxis.set_major_locator(major_locator)
        self.ax.yaxis.set_major_formatter(major_formatter)
        self.ax.yaxis.set_minor_locator(minor_locator)

        for key in ['800', '800pe', '900', '900pe',
                    '1000', '1000pe', '1100', '1100pe']:
            std = df.loc[df['key'] == key, 'charge'].std()
            self.log.info("ADC2PE {} stddev = {}".format(key, std))


class TMSpreadPlotter(OfficialPlotter):
    name = 'TMSpreadPlotter'

    def __init__(self, config, tool, **kwargs):
        """
        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, tool=tool, **kwargs)

    def create(self, df):
        keys = ['1100', '1000gm', '1000', '900', '800']
        for key in keys:
            means = np.zeros(32)
            for tm in range(32):
                means[tm] = df.loc[(df['key'] == key) & (df['tm'] == tm),
                                   'charge'].mean()
            if '800' in key:
                c = 'red'
            elif '900' in key:
                c = 'blue'
            elif '1000' in key:
                c = 'green'
            else:
                c = 'black'
            m = 'd' if 'gm' in key else 'o'
            self.ax.plot(np.arange(32), means, linestyle='-',
                         marker=m, color=c, label=key)
        self.ax.set_xlabel('TM')
        self.ax.set_ylabel('Charge (ADC)')
        self.ax.set_title('Mean Charge Across TMs')
        self.ax.legend(loc='upper right', prop={'size': 10})

        major_locator = MultipleLocator(5)
        major_formatter = FormatStrFormatter('%d')
        minor_locator = MultipleLocator(1)
        self.ax.xaxis.set_major_locator(major_locator)
        self.ax.xaxis.set_major_formatter(major_formatter)
        self.ax.xaxis.set_minor_locator(minor_locator)
        minor_locator = MultipleLocator(200)
        self.ax.yaxis.set_minor_locator(minor_locator)


class GMDistributionPlotter(OfficialPlotter):
    name = 'GMDistributionPlotter'

    def __init__(self, config, tool, **kwargs):
        """
        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, tool=tool, **kwargs)

    def create(self, df):
        charge_1000 = df.loc[df['key'] == '1000', 'charge']
        charge_1000pe = df.loc[df['key'] == '1000pe', 'charge']
        charge_1000gm = df.loc[df['key'] == '1000gm', 'charge']
        charge_1000gmpe = df.loc[df['key'] == '1000gmpe', 'charge']
        std_1000 = charge_1000.std()
        std_1000pe = charge_1000pe.std()
        std_1000gm = charge_1000gm.std()
        std_1000gmpe = charge_1000gmpe.std()
        median_1000 = np.median(charge_1000)
        median_1000pe = np.median(charge_1000pe)
        median_1000gm = np.median(charge_1000gm)
        median_1000gmpe = np.median(charge_1000gmpe)
        q75_1000pe, q25_1000pe = np.percentile(charge_1000pe, [75, 25])
        q75_1000, q25_1000 = np.percentile(charge_1000, [75, 25])
        # for tm in range(32):
        #     vals = df.loc[(df['key'] == '1000gmpe') & (df['tm'] == tm), 'charge']
        #     sns.kdeplot(vals, ax=self.ax, color="black", alpha=0.2, legend=False)
        sns.kdeplot(charge_1000, ax=self.ax, color="blue", shade=True, label='1000V (stddev = {:.2f})'.format(std_1000))
        sns.kdeplot(charge_1000pe, ax=self.ax, color="green", shade=True, label='1000V, ADC2PE Calibrated (stddev = {:.2f})'.format(std_1000pe))
        # sns.kdeplot(charge_1000gm, ax=self.ax, color="green", shade=True, label='GM avg=1000V (stddev = {:.2f})'.format(std_1000gm))
        # sns.kdeplot(charge_1000gmpe, ax=self.ax, color="blue", shade=True, label='GM avg=1000V, ADC2PE Calibrated (stddev = {:.2f})'.format(std_1000gmpe))

        x, y = self.ax.get_lines()[0].get_data()
        y_median_1000 = y[np.abs(x-median_1000).argmin()]
        y_q25_1000 = y[np.abs(x-q25_1000).argmin()]
        y_q75_1000 = y[np.abs(x-q75_1000).argmin()]
        x, y = self.ax.get_lines()[1].get_data()
        y_median_1000pe = y[np.abs(x-median_1000pe).argmin()]
        y_q25_1000pe = y[np.abs(x-q25_1000pe).argmin()]
        y_q75_1000pe = y[np.abs(x-q75_1000pe).argmin()]

        self.ax.vlines(median_1000, 0, y_median_1000, color="blue", linestyle='--')
        self.ax.vlines(q25_1000, 0, y_q25_1000, color="blue", linestyle=':')
        self.ax.vlines(q75_1000, 0, y_q75_1000, color="blue", linestyle=':')
        self.ax.vlines(median_1000pe, 0, y_median_1000pe, color="green", linestyle='--')
        self.ax.vlines(q25_1000pe, 0, y_q25_1000pe, color="green", linestyle=':')
        self.ax.vlines(q75_1000pe, 0, y_q75_1000pe, color="green", linestyle=':')

        self.ax.set_title("Distribution of Charge Across the Camera")
        self.ax.set_xlabel('Charge (p.e.)')
        self.ax.set_ylabel('Density')
        self.ax.legend(loc="upper right", prop={'size': 9})

        majorLocator = MultipleLocator(50)
        majorFormatter = FormatStrFormatter('%d')
        minorLocator = MultipleLocator(10)
        self.ax.xaxis.set_major_locator(majorLocator)
        self.ax.xaxis.set_major_formatter(majorFormatter)
        self.ax.xaxis.set_minor_locator(minorLocator)


class ADC2PEPlots(Tool):
    name = "ADC2PEPlots"
    description = "Create plots related to adc2pe"

    aliases = Dict(dict(max_events='TargetioFileReader.max_events'
                        ))
    classes = List([TargetioFileReader,
                    TargetioR1Calibrator,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reader_dict = dict()
        self.dl0 = None
        self.dl1 = None
        self.fitter = None
        self.dead = None

        self.n_pixels = None
        self.n_samples = None

        self.event_path_dict = {
            "800": "/Volumes/gct-jason/data/170310/hv/Run00904_r1_adc.tio",
            "800pe": "/Volumes/gct-jason/data/170310/hv/Run00904_r1.tio",
            "900": "/Volumes/gct-jason/data/170310/hv/Run00914_r1_adc.tio",
            "900pe": "/Volumes/gct-jason/data/170310/hv/Run00914_r1.tio",
            "1000": "/Volumes/gct-jason/data/170310/hv/Run00924_r1_adc.tio",
            "1000pe": "/Volumes/gct-jason/data/170310/hv/Run00924_r1.tio",
            "1100": "/Volumes/gct-jason/data/170310/hv/Run00934_r1_adc.tio",
            "1100pe": "/Volumes/gct-jason/data/170310/hv/Run00934_r1.tio",
            "1000gm": "/Volumes/gct-jason/data/170320/linearity/Run04174_r1_adc.tio",
            "1000gmpe": "/Volumes/gct-jason/data/170320/linearity/Run04174_r1.tio"
        }

        self.p_comparison = None
        self.p_tmspread = None
        self.p_dist = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        for key, path in self.event_path_dict.items():
            self.reader_dict[key] = TargetioFileReader(input_path=path, **kwargs)

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

        script = "checm_paper_adc2pe_application"
        self.p_comparison = ADC2PEPlotter(**kwargs, script=script, figure_name="adc2pe_comparison", shape='wide')
        self.p_tmspread = TMSpreadPlotter(**kwargs, script=script, figure_name="tmspread", shape='wide')
        self.p_dist = GMDistributionPlotter(**kwargs, script=script, figure_name="gm_distribution", shape='wide')

    def start(self):
        df_list = []

        desc1 = 'Looping through files'
        for key, reader in tqdm(self.reader_dict.items(), desc=desc1):
            hv = int(key.replace("pe", "").replace("gm", ""))
            cal = 'pe' in key
            gm = 'gm' in key
            cal_t = 'Calibrated' if 'pe' in key else 'Uncalibrated'

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

            desc3 = "Fitting Pixels"
            for pix in trange(self.n_pixels, desc=desc3):
                pixel_area = dl1[:, pix]
                if pix in self.dead.dead_pixels:
                    continue
                # if not self.fitter.apply(pixel_area):
                #     self.log.warning("File {} Pixel {} could not be fitted"
                #                      .format(key, pix))
                #     continue
                # charge = self.fitter.coeff['mean']
                charge = np.median(pixel_area)
                df_list.append(dict(key=key, hv=hv, cal=cal, cal_t=cal_t,
                                    gm=gm, pixel=pix, tm=pix//64,
                                    charge=charge))

        df = pd.DataFrame(df_list)

        # Obtain scale for different brigtness levels
        charge_tm9_1000gm = df.loc[(df['key'] == '1000gm') & (df['tm'] == 9), 'charge']
        charge_tm9_1100 = df.loc[(df['key'] == '1100') & (df['tm'] == 9), 'charge']
        brightness_ratio = np.median(charge_tm9_1100) / np.median(charge_tm9_1000gm)
        df.loc[df['key'] == '1000gm', 'charge'] *= brightness_ratio
        df.loc[df['key'] == '1000gmpe', 'charge'] *= brightness_ratio
        df_tmspread = df.copy()

        # Scale ADC values to match p.e.
        for key in ['800', '900', '1000', '1100', '1000gm']:
            pe_key = key + 'pe'
            df_hv = df.loc[(df['key'] == key) | (df['key'] == pe_key)]
            median_cal = np.median(df_hv.loc[df['cal']]['charge'])
            median_uncal = np.median(df_hv.loc[~df['cal']]['charge'])
            ratio = median_cal / median_uncal
            print(key, ratio)
            df.loc[(df['key'] == key) & ~df['cal'], 'charge'] *= ratio

        # where_gm = df['gm'] == True
        # df.loc[where_gm, 'charge'] /= tm9_1000gm.mean()
        # df.loc[~where_gm, 'charge'] /= tm9_1100.mean()

        df = df.sort_values(by=['cal', 'hv'], ascending=[True, True])

        # Create figures
        self.p_comparison.create(df.loc[~df['gm']])
        self.p_tmspread.create(df_tmspread.loc[~df_tmspread['cal']])
        self.p_dist.create(df)

    def finish(self):
        # Save figures
        self.p_comparison.save()
        self.p_tmspread.save()
        self.p_dist.save()


if __name__ == '__main__':
    exe = ADC2PEPlots()
    exe.run()
