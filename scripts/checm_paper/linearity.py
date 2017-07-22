from tqdm import tqdm, trange
from traitlets import Dict, List
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, \
    AutoMinorLocator, ScalarFormatter
import seaborn as sns
from scipy.stats import norm

from os.path import exists, join
from os import makedirs

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import AverageWfPeakIntegrator
from ctapipe.image.waveform_cleaning import CHECMWaveformCleanerAverage
from ctapipe.visualization import CameraDisplay
from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.calib.camera.r1 import TargetioR1Calibrator
from targetpipe.fitting.chec import CHECBrightFitter, CHECMSPEFitter
from targetpipe.calib.camera.adc2pe import TargetioADC2PECalibrator
from targetpipe.plots.official import OfficialPlotter
from targetpipe.io.pixels import Dead, get_geometry

from IPython import embed



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
        sns.violinplot(ax=self.ax, data=df, x='fw', y='illumination', hue='cal_t',
                       split=True, scale='count', inner='quartile',
                       legend=False)
        self.ax.set_title("Distribution Before and After ADC2PE Correction")
        self.ax.set_xlabel('FW')
        self.ax.set_ylabel('Illumination (p.e.)')
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
                means[tm] = df.loc[(df['key'] == key) & (df['tm'] == tm), 'illumination'].mean()
            self.ax.plot(np.arange(32), means, linestyle='-', marker='o', label=key)
        self.ax.set_xlabel('TM')
        self.ax.set_ylabel('Illumination (ADC)')
        self.ax.set_title('Mean Illumination Across TMs')
        self.ax.legend(loc='upper right', prop={'size': 10})

        major_locator = MultipleLocator(5)
        major_formatter = FormatStrFormatter('%d')
        minor_locator = MultipleLocator(1)
        self.ax.xaxis.set_major_locator(major_locator)
        self.ax.xaxis.set_major_formatter(major_formatter)
        self.ax.xaxis.set_minor_locator(minor_locator)
        minor_locator = MultipleLocator(200)
        self.ax.yaxis.set_minor_locator(minor_locator)


class Dist1D(OfficialPlotter):
    name = 'Dist1D'

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
        illumination = df.loc[df['key'] == '4050', 'illumination']
        illumination_pe = df.loc[df['key'] == '4050pe', 'illumination']
        std = illumination.std()
        std_pe = illumination_pe.std()
        median = np.median(illumination)
        median_pe = np.median(illumination_pe)
        q75, q25 = np.percentile(illumination, [75, 25])
        q75_pe, q25_pe = np.percentile(illumination_pe, [75, 25])
        sns.kdeplot(illumination, ax=self.ax, color="blue", shade=True, label='Uncal (stddev = {:.2f})'.format(std))
        sns.kdeplot(illumination_pe, ax=self.ax, color="green", shade=True, label='Cal, ADC2PE Calibrated (stddev = {:.2f})'.format(std_pe))

        x, y = self.ax.get_lines()[0].get_data()
        y_median_1000 = y[np.abs(x-median).argmin()]
        y_q25_1000 = y[np.abs(x-q25).argmin()]
        y_q75_1000 = y[np.abs(x-q75).argmin()]
        x, y = self.ax.get_lines()[1].get_data()
        y_median_1000pe = y[np.abs(x-median_pe).argmin()]
        y_q25_1000pe = y[np.abs(x-q25_pe).argmin()]
        y_q75_1000pe = y[np.abs(x-q75_pe).argmin()]

        self.ax.vlines(median, 0, y_median_1000, color="blue", linestyle='--')
        self.ax.vlines(q25, 0, y_q25_1000, color="blue", linestyle=':')
        self.ax.vlines(q75, 0, y_q75_1000, color="blue", linestyle=':')
        self.ax.vlines(median_pe, 0, y_median_1000pe, color="green", linestyle='--')
        self.ax.vlines(q25_pe, 0, y_q25_1000pe, color="green", linestyle=':')
        self.ax.vlines(q75_pe, 0, y_q75_1000pe, color="green", linestyle=':')

        self.ax.set_title("Distribution of Illumination Across the Camera")
        self.ax.set_xlabel('Illumination (p.e.)')
        self.ax.set_ylabel('Density')
        self.ax.legend(loc="upper right", prop={'size': 9})

        majorLocator = MultipleLocator(50)
        majorFormatter = FormatStrFormatter('%d')
        minorLocator = MultipleLocator(10)
        self.ax.xaxis.set_major_locator(majorLocator)
        self.ax.xaxis.set_major_formatter(majorFormatter)
        self.ax.xaxis.set_minor_locator(minorLocator)


class ImagePlotter(OfficialPlotter):
    name = 'ImagePlotter'

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

        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(1, 1, 1)

    def create(self, image, label, title):
        camera = CameraDisplay(get_geometry(), ax=self.ax,
                               image=image,
                               cmap='viridis')
        camera.add_colorbar()
        camera.colorbar.set_label(label, fontsize=20)
        camera.image = image
        camera.colorbar.ax.tick_params(labelsize=30)

        self.ax.set_title(title)
        self.ax.axis('off')


class Scatter(OfficialPlotter):
    name = 'Scatter'

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

        # self.fig = plt.figure(figsize=(12, 8))
        # self.ax = self.fig.add_subplot(1, 1, 1)

    def create(self, x, y, y_err, x_label="", y_label="", title=""):
        no_err = y_err == 0
        err = ~no_err
        self.ax.errorbar(x[no_err], y[no_err], fmt='o', mew=0.5, color='black', alpha=0.8, markersize=3, capsize=3)
        (_, caps, _) = self.ax.errorbar(x[err], y[err], yerr=y_err[err], fmt='o', mew=0.5, color='black', alpha=0.8, markersize=3, capsize=3)

        for cap in caps:
            cap.set_markeredgewidth(1)

        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xticks(x)
        self.ax.get_xaxis().set_major_formatter(ScalarFormatter())
        self.ax.get_yaxis().set_major_formatter(ScalarFormatter())
        self.ax.xaxis.set_tick_params(
            which='minor',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
        self.ax.xaxis.set_tick_params(which='major', labelsize=6.5)

        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.fig.suptitle(title)
        # self.ax.xaxis.set_major_locator(AutoMinorLocator(5))
        # self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))

        # axes[1].xaxis.set_minor_locator(AutoMinorLocator(5))
        # axes[2].yaxis.set_minor_locator(AutoMinorLocator(5))


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
        self.dead = None

        self.n_pixels = None
        self.n_samples = None

        self.df_file = None

        self.p_comparison = None
        self.p_tmspread = None
        self.p_dist = None
        self.p_image_saturated = None
        self.p_scatter_pix = None
        self.p_scatter_tm = None
        self.p_scatter_camera = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        fitter_4160 = CHECMSPEFitter(**kwargs)
        fitter_4160.range = [-30, 160]
        fitter_4161 = CHECMSPEFitter(**kwargs)
        fitter_4161.range = [-30, 160]
        fitter_4162 = CHECMSPEFitter(**kwargs)
        fitter_4162.range = [-30, 160]
        fitter_4160_pe = CHECMSPEFitter(**kwargs)
        fitter_4160_pe.range = [-3, 10]
        fitter_4160_pe.initial = dict(norm=None, eped=0, eped_sigma=0.2, spe=1,
                                      spe_sigma=0.5, lambda_=0.5)
        fitter_4161_pe = CHECMSPEFitter(**kwargs)
        fitter_4161_pe.range = [-3, 10]
        fitter_4161_pe.initial = dict(norm=None, eped=0, eped_sigma=0.2, spe=1,
                                      spe_sigma=0.5, lambda_=0.7)
        fitter_4162_pe = CHECMSPEFitter(**kwargs)
        fitter_4162_pe.range = [-3, 12]
        fitter_4162_pe.initial = dict(norm=None, eped=0, eped_sigma=0.2, spe=1,
                                      spe_sigma=0.5, lambda_=1)
        fitter_bright = CHECBrightFitter(**kwargs)

        dfl = []
        base_path = "/Volumes/gct-jason/data/170320/linearity/Run{:05}_r1_adc.tio"
        base_path_pe = "/Volumes/gct-jason/data/170320/linearity/Run{:05}_r1.tio"
        dfl.append(dict(path=base_path.format(4160), cal=False, fw=1250, fitter=fitter_4160))
        dfl.append(dict(path=base_path.format(4161), cal=False, fw=1450, fitter=fitter_4161))
        dfl.append(dict(path=base_path.format(4162), cal=False, fw=1650, fitter=fitter_4162))
        dfl.append(dict(path=base_path.format(4163), cal=False, fw=1850, fitter=fitter_bright))
        dfl.append(dict(path=base_path.format(4164), cal=False, fw=2050, fitter=fitter_bright))
        dfl.append(dict(path=base_path.format(4165), cal=False, fw=2250, fitter=fitter_bright))
        dfl.append(dict(path=base_path.format(4166), cal=False, fw=2450, fitter=fitter_bright))
        dfl.append(dict(path=base_path.format(4167), cal=False, fw=2650, fitter=fitter_bright))
        dfl.append(dict(path=base_path.format(4168), cal=False, fw=2850, fitter=fitter_bright))
        dfl.append(dict(path=base_path.format(4169), cal=False, fw=3050, fitter=fitter_bright))
        dfl.append(dict(path=base_path.format(4170), cal=False, fw=3250, fitter=fitter_bright))
        dfl.append(dict(path=base_path.format(4171), cal=False, fw=3450, fitter=fitter_bright))
        dfl.append(dict(path=base_path.format(4172), cal=False, fw=3650, fitter=fitter_bright))
        dfl.append(dict(path=base_path.format(4173), cal=False, fw=3850, fitter=fitter_bright))
        dfl.append(dict(path=base_path.format(4174), cal=False, fw=4050, fitter=fitter_bright))
        dfl.append(dict(path=base_path_pe.format(4160), cal=True, fw=1250, fitter=fitter_4160_pe))
        dfl.append(dict(path=base_path_pe.format(4161), cal=True, fw=1450, fitter=fitter_4161_pe))
        dfl.append(dict(path=base_path_pe.format(4162), cal=True, fw=1650, fitter=fitter_4162_pe))
        dfl.append(dict(path=base_path_pe.format(4163), cal=True, fw=1850, fitter=fitter_bright))
        dfl.append(dict(path=base_path_pe.format(4164), cal=True, fw=2050, fitter=fitter_bright))
        dfl.append(dict(path=base_path_pe.format(4165), cal=True, fw=2250, fitter=fitter_bright))
        dfl.append(dict(path=base_path_pe.format(4166), cal=True, fw=2450, fitter=fitter_bright))
        dfl.append(dict(path=base_path_pe.format(4167), cal=True, fw=2650, fitter=fitter_bright))
        dfl.append(dict(path=base_path_pe.format(4168), cal=True, fw=2850, fitter=fitter_bright))
        dfl.append(dict(path=base_path_pe.format(4169), cal=True, fw=3050, fitter=fitter_bright))
        dfl.append(dict(path=base_path_pe.format(4170), cal=True, fw=3250, fitter=fitter_bright))
        dfl.append(dict(path=base_path_pe.format(4171), cal=True, fw=3450, fitter=fitter_bright))
        dfl.append(dict(path=base_path_pe.format(4172), cal=True, fw=3650, fitter=fitter_bright))
        dfl.append(dict(path=base_path_pe.format(4173), cal=True, fw=3850, fitter=fitter_bright))
        dfl.append(dict(path=base_path_pe.format(4174), cal=True, fw=4050, fitter=fitter_bright))
        for d in dfl:
            d['reader'] = TargetioFileReader(input_path=d['path'], **kwargs)
        self.df_file = pd.DataFrame(dfl)

        cleaner = CHECMWaveformCleanerAverage(**kwargs)
        extractor = AverageWfPeakIntegrator(**kwargs)
        self.dl0 = CameraDL0Reducer(**kwargs)
        self.dl1 = CameraDL1Calibrator(extractor=extractor,
                                       cleaner=cleaner,
                                       **kwargs)
        self.dead = Dead()

        first_event = dfl[0]['reader'].get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        r1 = first_event.r1.tel[telid].pe_samples[0]
        self.n_pixels, self.n_samples = r1.shape

        script = "checm_paper_linearity"
        self.p_comparison = ViolinPlotter(**kwargs, script=script, figure_name="comparison", shape='wide')
        self.p_tmspread = TMSpreadPlotter(**kwargs, script=script, figure_name="tmspread", shape='wide')
        self.p_dist = Dist1D(**kwargs, script=script, figure_name="4050_distribution", shape='wide')
        self.p_image_saturated = ImagePlotter(**kwargs, script=script, figure_name="image_saturated")
        self.p_scatter_pix = Scatter(**kwargs, script=script, figure_name="scatter_pix", shape='wide')
        self.p_scatter_tm = Scatter(**kwargs, script=script, figure_name="scatter_tm", shape='wide')
        self.p_scatter_camera = Scatter(**kwargs, script=script, figure_name="scatter_camera", shape='wide')

    def start(self):
        # df_list = []
        #
        # desc1 = 'Looping through files'
        # n_rows = len(self.df_file.index)
        # for index, row in tqdm(self.df_file.iterrows(), total=n_rows, desc=desc1):
        #     path = row['path']
        #     reader = row['reader']
        #     cal = row['cal']
        #     fw = row['fw']
        #     fitter = row['fitter']
        #
        #     cal_t = 'Calibrated' if cal else 'Uncalibrated'
        #     key = '{}pe'.format(fw) if cal else '{}'.format(fw)
        #
        #     source = reader.read()
        #     n_events = reader.num_events
        #
        #     dl1 = np.zeros((n_events, self.n_pixels))
        #
        #     desc2 = "Extracting Charge"
        #     for event in tqdm(source, desc=desc2, total=n_events):
        #         ev = event.count
        #         self.dl0.reduce(event)
        #         self.dl1.calibrate(event)
        #         dl1[ev] = event.dl1.tel[0].image[0]
        #
        #     desc3 = "Fitting Pixels"
        #     for pix in trange(self.n_pixels, desc=desc3):
        #         pixel_area = dl1[:, pix]
        #         if pix in self.dead.dead_pixels:
        #             continue
        #         if not fitter.apply(pixel_area):
        #             self.log.warning("File {} Pixel {} could not be fitted"
        #                              .format(path, pix))
        #             continue
        #         if fitter.fitter_type == 'spe':
        #             illumination = fitter.coeff['lambda_']
        #             illumination_err = 0
        #         else:
        #             illumination = fitter.coeff['mean']
        #             illumination_err = fitter.coeff['stddev']
        #
        #         # illumination = np.median(pixel_area)
        #         # q75, q25 = np.percentile(pixel_area, [75, 25])
        #         # illumination_err = q75 - q25
        #         df_list.append(dict(key=key, fw=fw, cal=cal, cal_t=cal_t,
        #                             pixel=pix, tm=pix//64,
        #                             illumination=illumination,
        #                             illumination_err=illumination_err))
        #
        # df = pd.DataFrame(df_list)
        # store = pd.HDFStore('/Users/Jason/Downloads/linearity.h5')
        # store['df'] = df

        store = pd.HDFStore('/Users/Jason/Downloads/linearity.h5')
        df = store['df']

        df = df.sort_values(by='cal', ascending=True)
        df_raw = df.copy()

        # Scale ADC values to match p.e.
        fw_list = np.unique(df['fw'])
        for fw in fw_list:
            df_fw = df.loc[df['fw'] == fw]
            median_cal = np.median(df_fw.loc[df['cal']]['illumination'])
            median_uncal = np.median(df_fw.loc[~df['cal']]['illumination'])
            ratio = median_cal / median_uncal
            df.loc[(df['fw'] == fw) & ~df['cal'], 'illumination'] *= ratio

        # Create figures
        self.p_comparison.create(df)
        self.p_tmspread.create(df_raw.loc[df_raw['cal']])
        self.p_dist.create(df)

        image = np.zeros(self.n_pixels)
        illumination = df.loc[df['key'] == '4050pe', 'illumination']
        pix = df.loc[df['key'] == '4050pe', 'pixel']
        image[pix] = illumination
        image = self.dead.mask1d(image)
        self.p_image_saturated.create(image, "Illumination (p.e.)", "Saturated Run")

        p = 537
        df_pix = df.loc[(df['pixel'] == p) & (df['cal'])]
        x = df_pix['fw']
        y = df_pix['illumination']
        y_err = df_pix['illumination_err']
        self.p_scatter_pix.create(x, y, y_err, "Filter Wheel", "Illumination (p.e.)", "Pixel {}".format(p))

        tm = 8
        df_tm = df.loc[(df['tm'] == tm) & (df['cal'])].groupby('fw')
        x = df_tm.apply(np.mean).index.values
        y = df_tm.apply(np.mean)['illumination']
        y_err = df_tm.apply(np.std)['illumination']
        self.p_scatter_tm.create(x, y, y_err, "Filter Wheel", "Illumination (p.e.)", "TM {}".format(tm))

        df_camera = df.loc[df['cal']].groupby('fw')
        x = df_camera.apply(np.mean).index.values
        y = df_camera.apply(np.mean)['illumination']
        y_err = df_camera.apply(np.std)['illumination']
        self.p_scatter_camera.create(x, y, y_err, "Filter Wheel", "Illumination (p.e.)", "Whole Camera")

    def finish(self):
        # Save figures
        self.p_comparison.save()
        self.p_tmspread.save()
        self.p_dist.save()
        self.p_image_saturated.save()
        self.p_scatter_pix.save()
        self.p_scatter_tm.save()
        self.p_scatter_camera.save()


if __name__ == '__main__':
    exe = ADC2PEPlots()
    exe.run()
