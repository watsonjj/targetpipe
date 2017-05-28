from tqdm import tqdm, trange
from traitlets import Dict, List
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
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
from targetpipe.fitting.checm import CHECBrightFitter
from targetpipe.io.pixels import Dead
from targetpipe.calib.camera.adc2pe import TargetioADC2PECalibrator
from targetpipe.plots.official import OfficialPlotter


class ViolinPlotter(OfficialPlotter):
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
        sns.violinplot(ax=self.ax, data=df, x='fw', y='charge', hue='cal_t',
                       split=True, scale='count', inner='quartile',
                       legend=False)
        self.ax.set_title("Distribution Before and After ADC2PE Correction")
        self.ax.set_xlabel('FW')
        self.ax.set_ylabel('Charge (p.e.)')
        self.ax.legend(loc="upper right")

        major_locator = MultipleLocator(50)
        major_formatter = FormatStrFormatter('%d')
        minor_locator = MultipleLocator(10)
        self.ax.yaxis.set_major_locator(major_locator)
        self.ax.yaxis.set_major_formatter(major_formatter)
        self.ax.yaxis.set_minor_locator(minor_locator)


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
        keys = np.unique(df['key'])
        for key in keys:
            means = np.zeros(32)
            for tm in range(32):
                means[tm] = df.loc[(df['key'] == key) & (df['tm'] == tm), 'charge'].mean()
            self.ax.plot(np.arange(32), means, linestyle='-', marker='o', label=key)
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
        std_1000pe = charge_1000pe.std()
        std_1000 = charge_1000.std()
        std_1000gm = charge_1000gm.std()
        std_1000gmpe = charge_1000gmpe.std()
        # for tm in range(32):
        #     vals = df.loc[(df['key'] == '1000gmpe') & (df['tm'] == tm), 'charge']
        #     sns.kdeplot(vals, ax=self.ax, color="black", alpha=0.2, legend=False)
        sns.kdeplot(charge_1000, ax=self.ax, color="red", shade=True, label='1000V (stddev = {:.2f})'.format(std_1000))
        sns.kdeplot(charge_1000pe, ax=self.ax, color="black", shade=True, label='1000V (stddev = {:.2f})'.format(std_1000pe))
        sns.kdeplot(charge_1000gm, ax=self.ax, color="green", shade=True, label='GM avg=1000V (stddev = {:.2f})'.format(std_1000gm))
        sns.kdeplot(charge_1000gmpe, ax=self.ax, color="blue", shade=True, label='GM avg=1000V, ADC2PE Calibrated (stddev = {:.2f})'.format(std_1000gmpe))
        self.ax.set_title("Distribution of Charge Across the Camera")
        self.ax.set_xlabel('Charge (p.e.)')
        self.ax.set_ylabel('Density')
        self.ax.legend(loc="upper right", prop={'size': 10})

        # majorLocator = MultipleLocator(10)
        # majorFormatter = FormatStrFormatter('%d')
        # minorLocator = MultipleLocator(2)
        # self.ax.xaxis.set_major_locator(majorLocator)
        # self.ax.xaxis.set_major_formatter(majorFormatter)
        # self.ax.xaxis.set_minor_locator(minorLocator)


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

        n_files = 15
        fw = np.linspace(1250, 4050, n_files)
        fw_start = 1250
        fw_step = 200
        run_start = 4160
        base_path = "/Volumes/gct-jason/data/170320/linearity/Run{:05}_r1_adc.tio"
        base_path_pe = "/Volumes/gct-jason/data/170320/linearity/Run{:05}_r1.tio"
        self.event_path_dict = dict()
        for i in range(n_files):
            fw = fw_start + fw_step * i
            run = run_start + i
            self.event_path_dict[str(fw)] = base_path.format(run)
            self.event_path_dict["{}pe".format(fw)] = base_path_pe.format(run)

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

        script = "checm_paper_linearity"
        self.p_comparison = ViolinPlotter(**kwargs, script=script, figure_name="comparison", shape='wide')
        self.p_tmspread = TMSpreadPlotter(**kwargs, script=script, figure_name="tmspread", shape='wide')
        # self.p_dist = GMDistributionPlotter(**kwargs, script=script, figure_name="gm_distribution", shape='wide')

    def start(self):
        df_list = []

        desc1 = 'Looping through files'
        for key, reader in tqdm(self.reader_dict.items(), desc=desc1):
            fw = int(key.replace("pe", ""))
            cal = 'pe' in key
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
                df_list.append(dict(key=key, fw=fw, cal=cal, cal_t=cal_t,
                                    pixel=pix, tm=pix//64,
                                    charge=charge))

        df = pd.DataFrame(df_list)
        df = df.sort_values(by='cal', ascending=True)
        df_raw = df.copy()

        # Scale ADC values to match p.e.
        fw_list = np.unique(df['fw'])
        for fw in fw_list:
            df_fw = df.loc[df['fw'] == fw]
            median_cal = np.median(df_fw.loc[df['cal']]['charge'])
            median_uncal = np.median(df_fw.loc[~df['cal']]['charge'])
            ratio = median_cal / median_uncal
            df.loc[(df['fw'] == fw) & ~df['cal'], 'charge'] *= ratio

        # Create figures
        self.p_comparison.create(df)
        self.p_tmspread.create(df_raw.loc[df_raw['cal']])
        # self.p_dist.create(df)

    def finish(self):
        # Save figures
        self.p_comparison.save()
        self.p_tmspread.save()
        # self.p_dist.save()


if __name__ == '__main__':
    exe = ADC2PEPlots()
    exe.run()
