from targetpipe.io.camera import Config
Config('checm')

from tqdm import tqdm, trange
from traitlets import Dict, List
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import seaborn as sns

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import AverageWfPeakIntegrator
from ctapipe.image.waveform_cleaning import CHECMWaveformCleanerAverage
from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.calib.camera.r1 import TargetioR1Calibrator
from targetpipe.fitting.chec import CHECMSPEFitter
from targetpipe.io.pixels import Dead
from targetpipe.calib.camera.adc2pe import TargetioADC2PECalibrator
from targetpipe.plots.official import ChecmPaperPlotter
from targetpipe.utils.dactov import checm_dac_to_volts


class PixelSPEFitPlotter(ChecmPaperPlotter):
    name = 'PixelSPEFitPlotter'

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

    def create(self, hist, edges, between, fit, fitx):
        # Normalise histogram
        norm = np.sum(np.diff(edges) * hist)
        hist_n = hist#/norm
        fit_n = fit#/norm

        # Roll axis for easier plotting
        nbins = hist_n.size
        hist_tops = np.insert(hist_n, np.arange(nbins), hist_n, axis=0)
        edges_tops = np.insert(edges, np.arange(edges.shape[0]),
                               edges, axis=0)[1:-1]

        self.ax.semilogy(edges_tops, hist_tops, color='black', alpha=0.5)
        self.ax.semilogy(fitx, fit_n, color='black')
        # self.ax.set_ylim(ymin=3e-2)
        # self.ax.set_title("SPE Spectrum, Pixel 1559")
        self.ax.set_xlabel("Pulse Area (V ns)")
        self.ax.set_ylabel("Counts")

        major_locator = MultipleLocator(0.05)
        # major_formatter = FormatStrFormatter('%d')
        minor_locator = MultipleLocator(0.01)
        self.ax.xaxis.set_major_locator(major_locator)
        # self.ax.xaxis.set_major_formatter(major_formatter)
        self.ax.xaxis.set_minor_locator(minor_locator)


class TMSPEFitPlotter(ChecmPaperPlotter):
    name = 'TMSPEFitPlotter'

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

    def create(self, hist, edges, between, x_unit, x_major=0.05):

        # Normalise histogram
        norm = np.sum(np.diff(edges, axis=1) * hist, axis=1)
        hist_n = hist/norm[:, None]

        # Roll axis for easier plotting
        hist_r = np.rollaxis(hist_n, 1)
        nbins, npix = hist_r.shape
        e = edges[0]
        hist_tops = np.insert(hist_r, np.arange(nbins), hist_r, axis=0)
        edges_tops = np.insert(e, np.arange(e.shape[0]), e, axis=0)[1:-1]

        self.ax.semilogy(edges_tops, hist_tops, color='black', alpha=0.2)
        # self.ax.set_ylim(ymin=1e-4)
        # self.ax.set_title("SPE Spectrum, TM 24")
        self.ax.set_xlabel("Pulse Area ({})".format(x_unit))
        self.ax.set_ylabel("Probability Density")

        major_locator = MultipleLocator(x_major)
        # major_formatter = FormatStrFormatter('%d')
        minor_locator = MultipleLocator(x_major/5)
        self.ax.xaxis.set_major_locator(major_locator)
        # self.ax.xaxis.set_major_formatter(major_formatter)
        self.ax.xaxis.set_minor_locator(minor_locator)


class ADC2PEPlotter(ChecmPaperPlotter):
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

        sns.violinplot(ax=self.ax, data=df, x='hv', y='spe_mv', hue='gm_t',
                       split=True, scale='count', inner='quartile',
                       legend=False)
        # self.ax.set_title("SPE Values for different HV Settings")
        self.ax.set_xlabel('HV (V)')
        self.ax.set_ylabel('SPE Value (V ns / p.e.)')
        self.ax.legend(loc="upper left")

        for key in ['800', '800gm', '900', '900gm', '1000', '1000gm', '1100']:
            std = df.loc[df['key'] == key, 'spe'].std()
            self.log.info("ADC2PE {}V stddev = {}".format(key, std))


class ADC2PE1100VTMPlotter(ChecmPaperPlotter):
    name = 'ADC2PE1100VTMPlotter'

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
        df_1100 = df[df['key'] == '1100']
        for tm in range(32):
            vals = df_1100.loc[df['tm'] == tm, 'spe_mv']
            sns.kdeplot(vals, ax=self.ax,
                        color="black", alpha=0.5, legend=False)
            self.log.info("ADC2PE 1100V TM{} mean = {:.4f}, "
                          "median = {:.4f}, stddev = {:.4f}"
                          .format(tm, np.mean(vals), np.median(vals),
                                  np.std(vals)))
        vals = df_1100['spe_mv']
        sns.kdeplot(vals, ax=self.ax, color="blue", legend=False, lw=3)
        # self.ax.set_title("Distribution of SPE Values Across "
        #                   "the Camera with HV=1100V")
        self.ax.set_xlabel('SPE Value (V*ns)')
        self.ax.set_ylabel('Density')

        black_line = mlines.Line2D([], [], color='black', alpha=0.5,
                                   label="Distribution within TM")
        blue_line = mlines.Line2D([], [], color='blue',
                                  label="Distribution across Camera")
        self.ax.legend(handles=[black_line, blue_line], loc="upper right")

        major_locator = MultipleLocator(0.01)
        # major_formatter = FormatStrFormatter('%d')
        minor_locator = MultipleLocator(0.002)
        self.ax.xaxis.set_major_locator(major_locator)
        # self.ax.xaxis.set_major_formatter(major_formatter)
        self.ax.xaxis.set_minor_locator(minor_locator)


class ADC2PE1100VTMStatsPlotter(ChecmPaperPlotter):
    name = 'ADC2PE1100VTMStatsPlotter'

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
        df_1100 = df[df['key'] == '1100']
        means = df_1100.groupby('tm')['spe_mv'].mean()
        stds = df_1100.groupby('tm')['spe_mv'].std()
        fracspreads = stds/means
        sns.distplot(fracspreads, ax=self.ax, color="black", rug=True)
        # self.ax.set_title("Fractional spread of SPE Value (V*ns) "
        #                   "between TMs at 1100V")
        self.ax.set_xlabel('Fractional spread')
        self.ax.set_ylabel('Density')

        major_locator = MultipleLocator(0.05)
        minor_locator = MultipleLocator(0.01)
        self.ax.xaxis.set_major_locator(major_locator)
        self.ax.xaxis.set_minor_locator(minor_locator)


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

        self.reader = None
        self.reader_pe = None
        self.dl0 = None
        self.dl1 = None
        self.fitter = None
        self.fitter_pe = None
        self.dead = None

        self.n_pixels = None
        self.n_samples = None

        self.spe_path = "/Volumes/gct-jason/data/170314/spe/Run00073_r0/extract_spe/spe.npy"
        self.gm_path = "/Volumes/gct-jason/data/170310/hv/gain_matching_coeff.npz"

        self.p_pixelspe = None
        self.p_tmspe = None
        self.p_tmspe_pe = None
        self.p_adc2pe = None
        self.p_adc2pe_1100tm = None
        self.p_adc2pe_1100tm_stats = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        filepath = '/Volumes/gct-jason/data/170314/spe/Run00073_r1_adc.tio'
        self.reader = TargetioFileReader(input_path=filepath, **kwargs)

        filepath = '/Volumes/gct-jason/data/170314/spe/Run00073_r1.tio'
        self.reader_pe = TargetioFileReader(input_path=filepath, **kwargs)

        cleaner = CHECMWaveformCleanerAverage(**kwargs)
        extractor = AverageWfPeakIntegrator(**kwargs)
        self.dl0 = CameraDL0Reducer(**kwargs)
        self.dl1 = CameraDL1Calibrator(extractor=extractor,
                                       cleaner=cleaner,
                                       **kwargs)
        self.fitter = CHECMSPEFitter(**kwargs)
        self.fitter.range = [-30, 160]
        self.fitter_pe = CHECMSPEFitter(**kwargs)
        self.fitter_pe.range = [-1, 6]
        self.fitter_pe.initial = dict(norm=None,
                                   eped=0,
                                   eped_sigma=0.2,
                                   spe=1,
                                   spe_sigma=0.5,
                                   lambda_=0.2)
        self.dead = Dead()

        script = "checm_paper_adc2pe"
        self.p_pixelspe = PixelSPEFitPlotter(**kwargs, script=script, figure_name="spe_fit_pixel1559")
        self.p_tmspe = TMSPEFitPlotter(**kwargs, script=script, figure_name="spe_fit_tm24")
        self.p_tmspe_pe = TMSPEFitPlotter(**kwargs, script=script, figure_name="spe_fit_tm24_pe")
        self.p_adc2pe = ADC2PEPlotter(**kwargs, script=script, figure_name="adc2pe", shape='square')
        self.p_adc2pe_1100tm = ADC2PE1100VTMPlotter(**kwargs, script=script, figure_name="adc2pe_1100V_tms", shape='wide')
        self.p_adc2pe_1100tm_stats = ADC2PE1100VTMStatsPlotter(**kwargs, script=script, figure_name="adc2pe_1100V_tms_stats", shape='wide')

    def start(self):
        n_events = self.reader.num_events
        first_event = self.reader.get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        n_pixels, n_samples = first_event.r1.tel[telid].pe_samples[0].shape

        ### SPE values from fit _______________________________________________
        # Prepare storage array

        dl1 = np.zeros((n_events, n_pixels))
        dl1_pe = np.zeros((n_events, n_pixels))
        hist_pix1559 = None
        edges_pix1559 = None
        between_pix1559 = None
        fit_pix1559 = None
        fitx_pix1559 = None
        hist_tm24 = np.zeros((64, self.fitter.nbins))
        edges_tm24 = np.zeros((64, self.fitter.nbins + 1))
        between_tm24 = np.zeros((64, self.fitter.nbins))
        hist_tm24_pe = np.zeros((64, self.fitter.nbins))
        edges_tm24_pe = np.zeros((64, self.fitter.nbins + 1))
        between_tm24_pe = np.zeros((64, self.fitter.nbins))

        source = self.reader.read()
        desc = "Looping through file"
        for event in tqdm(source, total=n_events, desc=desc):
            index = event.count
            self.dl0.reduce(event)
            self.dl1.calibrate(event)
            dl1[index] = event.dl1.tel[telid].image

        source = self.reader_pe.read()
        desc = "Looping through file (pe)"
        for event in tqdm(source, total=n_events, desc=desc):
            index = event.count
            self.dl0.reduce(event)
            self.dl1.calibrate(event)
            dl1_pe[index] = event.dl1.tel[telid].image

        desc = "Fitting pixels"
        for pix in trange(n_pixels, desc=desc):
            tm = pix // 64
            tmpix = pix % 64
            if tm != 24:
                continue
            if not self.fitter.apply(dl1[:, pix]):
                self.log.warning("Pixel {} couldn't be fit".format(pix))
                continue
            if pix == 1559:
                hist_pix1559 = self.fitter.hist
                edges_pix1559 = self.fitter.edges
                between_pix1559 = self.fitter.between
                fit_pix1559 = self.fitter.fit
                fitx_pix1559 = self.fitter.fit_x
            hist_tm24[tmpix] = self.fitter.hist
            edges_tm24[tmpix] = self.fitter.edges
            between_tm24[tmpix] = self.fitter.between

        edges_pix1559 = checm_dac_to_volts(edges_pix1559)
        between_pix1559 = checm_dac_to_volts(between_pix1559)
        fitx_pix1559 = checm_dac_to_volts(fitx_pix1559)
        edges_tm24 = checm_dac_to_volts(edges_tm24)
        between_tm24 = checm_dac_to_volts(edges_tm24)

        desc = "Fitting pixels (pe)"
        for pix in trange(n_pixels, desc=desc):
            tm = pix // 64
            tmpix = pix % 64
            if tm != 24:
                continue
            if not self.fitter_pe.apply(dl1_pe[:, pix]):
                self.log.warning("Pixel {} couldn't be fit".format(pix))
                continue
            hist_tm24_pe[tmpix] = self.fitter_pe.hist
            edges_tm24_pe[tmpix] = self.fitter_pe.edges
            between_tm24_pe[tmpix] = self.fitter_pe.between

        ### SPE values for each hv setting ____________________________________
        kwargs = dict(config=self.config, tool=self,
                      spe_path=self.spe_path, gain_matching_path=self.gm_path)
        a2p = TargetioADC2PECalibrator(**kwargs)
        hv_dict = dict()
        hv_dict['800'] = [800] * 2048
        hv_dict['900'] = [900] * 2048
        hv_dict['1000'] = [1000] * 2048
        hv_dict['1100'] = [1100] * 2048
        hv_dict['800gm'] = [a2p.gm800[i//64] for i in range(2048)]
        hv_dict['900gm'] = [a2p.gm900[i//64] for i in range(2048)]
        hv_dict['1000gm'] = [a2p.gm1000[i//64] for i in range(2048)]
        df_list = []
        for key, l in hv_dict.items():
            hv = int(key.replace("gm", ""))
            gm = 'gm' in key
            gm_t = 'Gain-matched' if 'gm' in key else 'Non-gain-matched'
            for pix in range(n_pixels):
                if pix in self.dead.dead_pixels:
                    continue
                adc2pe = a2p.get_adc2pe_at_hv(l[pix], pix)
                df_list.append(dict(key=key, hv=hv, gm=gm, gm_t=gm_t,
                                    pixel=pix, tm=pix//64, spe=1/adc2pe))

        df = pd.DataFrame(df_list)
        df = df.sort_values(by='gm', ascending=True)
        df = df.assign(spe_mv=checm_dac_to_volts(df['spe']))

        # Create figures
        self.p_pixelspe.create(hist_pix1559, edges_pix1559, between_pix1559,
                               fit_pix1559, fitx_pix1559)
        self.p_tmspe.create(hist_tm24, edges_tm24, between_tm24, "V*ns")
        self.p_tmspe_pe.create(hist_tm24_pe, edges_tm24_pe, between_tm24_pe,
                               "p.e.", 1)
        self.p_adc2pe.create(df)
        self.p_adc2pe_1100tm.create(df)
        self.p_adc2pe_1100tm_stats.create(df)

    def finish(self):
        # Save figures
        self.p_pixelspe.save()
        self.p_tmspe.save()
        self.p_tmspe_pe.save()
        self.p_adc2pe.save()
        self.p_adc2pe_1100tm.save()
        self.p_adc2pe_1100tm_stats.save()


if __name__ == '__main__':
    exe = ADC2PEPlots()
    exe.run()
