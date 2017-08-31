from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.signal import general_gaussian
from tqdm import tqdm
from traitlets import Dict, List
from matplotlib import pyplot as plt
import numpy as np
from os.path import exists, join
from os import makedirs
from IPython import embed
import pandas as pd
from scipy import interpolate
from scipy.ndimage import correlate1d

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.core import Tool, Component
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from ctapipe.image.waveform_cleaning import WaveformCleanerFactory
from ctapipe.io.eventfilereader import EventFileReaderFactory
from targetpipe.utils.correlate import cross_correlate
from targetpipe.fitting.chec import CHECSSPEFitter


class CleanerNull(Component):
    name = "CleanerNull"

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

    def clean(self, waveforms, connected):
        return np.copy(waveforms)


class CleanerGaussConvolve(Component):
    name = "CleanerGaussConvolve"

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        self.kernel = general_gaussian(5, p=1.0, sig=1)

    def clean(self, waveforms, connected):
        smooth_flat = np.convolve(waveforms.ravel(), self.kernel, "same")
        smoothed = np.reshape(smooth_flat, waveforms.shape)
        samples_std = np.std(waveforms, axis=1)
        smooth_baseline_std = np.std(smoothed, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            smoothed *= (samples_std / smooth_baseline_std)[:, None]
            smoothed[~np.isfinite(smoothed)] = 0
        return smoothed


class CleanerCrossCorrelate(Component):
    name = "CleanerCrossCorrelate"

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        file = np.loadtxt("/Users/Jason/Downloads/pulse_data.txt", delimiter=', ')
        refx = file[:, 0]
        refy = file[:, 1] - file[:, 1][0]
        f = interpolate.interp1d(refx, refy, kind=3)
        x = np.linspace(0, 77e-9, 76)
        y = f(x)
        self.reference_pulse = y

    def clean(self, waveforms, connected):
        return self.clean2(waveforms, connected)

    def clean1(self, waveforms, connected):
        smooth_flat = np.correlate(waveforms.ravel(), self.reference_pulse, "same")
        smoothed = np.reshape(smooth_flat, waveforms.shape)
        # samples_sum = np.sum(waveforms)
        # smooth_baseline_sum = np.sum(smoothed)
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     smoothed = smoothed * (samples_sum / smooth_baseline_sum)
        #     smoothed[~np.isfinite(smoothed)] = 0
        return smoothed

    def clean2(self, waveforms, connected):
        smoothed = correlate1d(waveforms, self.reference_pulse)
        # samples_sum = np.sum(waveforms, axis=1)
        # s_sum = np.sum(smoothed, axis=1)
        # smoothed *= (samples_sum / s_sum)[:, None]
        return smoothed


class CleanerCrossCorrelateC(Component):
    name = "CleanerCrossCorrelateC"

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        file = np.loadtxt("/Users/Jason/Downloads/pulse_data.txt", delimiter=', ')
        refx = file[:, 0]
        refy = file[:, 1] - file[:, 1][0]
        f = interpolate.interp1d(refx, refy, kind=3)
        x = np.linspace(0, 77e-9, 76)
        y = f(x)
        self.reference_pulse = y

    def clean(self, waveforms, connected):
        smoothed = cross_correlate(waveforms, self.reference_pulse)
        samples_sum = np.sum(waveforms, axis=1)
        s_sum = np.sum(smoothed, axis=1)
        smoothed *= (samples_sum / s_sum)[:, None]
        return smoothed


class CleanerSmoothBaselineSubtract(Component):
    name = "CleanerSmoothBaselineSubtract"

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        self.kernel = general_gaussian(10, p=1.0, sig=32)

    def clean(self, waveforms, connected):
        # Subtract initial baseline
        baseline_sub = waveforms - np.mean(waveforms[:, :32], axis=1)[:, None]

        # Obtain waveform with pulse masked
        avgwf = np.mean(waveforms[connected], 0)
        t0 = np.argmax(avgwf)
        mask = np.zeros(waveforms.shape, dtype=np.bool)
        mask[:, t0-10:t0+10] = True
        masked = np.ma.masked_array(baseline_sub, mask)
        no_pulse = np.ma.filled(masked, 0)

        # Get smooth baseline (no pulse)
        smooth_flat = np.convolve(no_pulse.ravel(), self.kernel, "same")
        smooth_baseline = np.reshape(smooth_flat, waveforms.shape)
        no_pulse_std = np.std(no_pulse, axis=1)
        smooth_baseline_std = np.std(smooth_baseline, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            smooth_baseline *= (no_pulse_std / smooth_baseline_std)[:, None]
            smooth_baseline[~np.isfinite(smooth_baseline)] = 0

        # Get smooth waveform
        smooth_wf = baseline_sub  # self.wf_smoother.apply(baseline_sub)

        # Subtract smooth baseline
        cleaned = smooth_wf - smooth_baseline
        return cleaned


class CleanerSBSGC(Component):
    name = "CleanerSBSGC"

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        self.sbs = CleanerSmoothBaselineSubtract(None, None)
        self.smooth = CleanerGaussConvolve(None, None)

    def clean(self, waveforms, connected):
        waveforms = self.smooth.clean(waveforms, connected)
        waveforms = self.sbs.clean(waveforms, connected)

        return waveforms


class SiPMSPETesting(Tool):
    name = "SiPMSPETesting"
    description = "Diagnosis script in order to find the spe spectrum of CHECS"

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        pe='CameraR1CalibratorFactory.pe_path',
                        extractor='ChargeExtractorFactory.extractor',
                        extractor_t0='ChargeExtractorFactory.t0',
                        window_width='ChargeExtractorFactory.window_width',
                        window_shift='ChargeExtractorFactory.window_shift',
                        sig_amp_cut_HG='ChargeExtractorFactory.sig_amp_cut_HG',
                        sig_amp_cut_LG='ChargeExtractorFactory.sig_amp_cut_LG',
                        lwt='ChargeExtractorFactory.lwt',
                        clip_amplitude='CameraDL1Calibrator.clip_amplitude',
                        radius='CameraDL1Calibrator.radius',
                        cleaner='WaveformCleanerFactory.cleaner',
                        cleaner_t0='WaveformCleanerFactory.t0',
                        ))
    classes = List([EventFileReaderFactory,
                    ChargeExtractorFactory,
                    CameraR1CalibratorFactory,
                    CameraDL1Calibrator,
                    WaveformCleanerFactory
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reader = None
        self.r1 = None
        self.fitter = None

        self.n_pixels = None
        self.n_samples = None

        self.connected = None
        self.match_t0 = None
        self.within = None
        self.ws = None
        self.we = None
        self.cleaners = None
        self.output_dir = None
        self.df = None
        self.dfwf = None

        self.poi = [22, 35, 40, 41, 42, 43]
        self.wf_poi = 35
        # self.eoi = 11
        self.eoi = 10

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.reader = reader_class(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin=self.reader.origin,
                                               **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)

        self.fitter = CHECSSPEFitter(**kwargs)

        first_event = self.reader.get_event(0)
        self.n_pixels = first_event.inst.num_pixels[0]
        self.n_samples = first_event.r0.tel[0].num_samples

        cleaners = [
            CleanerNull(**kwargs),
            CleanerGaussConvolve(**kwargs),
            # CleanerSmoothBaselineSubtract(**kwargs),
            # CleanerSBSGC(**kwargs),
            CleanerCrossCorrelate(**kwargs),
            # CleanerCrossCorrelateC(**kwargs),
        ]
        self.cleaners = {i.name: i for i in cleaners}

        self.output_dir = join(self.reader.output_directory, "sipm_spe_testing")

        # connected = [40, 41, 42, 43]
        connected = list(range(self.n_pixels))
        ignore_pixels = [25, 26, 18, 19, 13, 14, 5, 6, 49, 37]
        self.connected = [i for i in connected if i not in ignore_pixels]
        shift = 4
        width = 8
        self.match_t0 = 60
        self.within = 3

        self.ws = self.match_t0 - shift
        if self.ws < 0:
            self.ws = 0
        self.we = self.ws + width
        if self.we > self.n_samples - 1:
            self.we = self.n_samples - 1

    def start(self):
        # recalculate = True
        recalculate = False
        if recalculate:
            n_events = self.reader.num_events
            source = self.reader.read()

            df_list = []
            dfwf_list = []

            a_event = np.indices((n_events, self.n_pixels))[0]
            a_pixel = np.indices((n_events, self.n_pixels))[1]
            a_row = dict()
            a_area = dict()
            a_height = dict()
            a_peakpos = dict()
            a_height_at_t0 = dict()
            a_height_in_window = dict()
            a_peakpos_in_window = dict()
            for c in self.cleaners.keys():
                a_row[c] = np.zeros((n_events, self.n_pixels))
                a_area[c] = np.zeros((n_events, self.n_pixels))
                a_height[c] = np.zeros((n_events, self.n_pixels))
                a_peakpos[c] = np.zeros((n_events, self.n_pixels))
                a_height_at_t0[c] = np.zeros((n_events, self.n_pixels))
                a_height_in_window[c] = np.zeros((n_events, self.n_pixels))
                a_peakpos_in_window[c] = np.zeros((n_events, self.n_pixels))

            desc = "Looping through file"
            for event in tqdm(source, desc=desc, total=n_events):
                ev = event.count
                row = event.r0.tel[0].row
                self.r1.calibrate(event)
                r1 = event.r1.tel[0].pe_samples[0]

                connected = self.connected
                match_t0 = self.match_t0
                ws = self.ws
                we = self.we

                for c, cleaner in self.cleaners.items():
                    r1c = cleaner.clean(r1, connected)

                    # Get average pulse of event
                    avgwf = np.mean(r1c[connected], 0)
                    avgwf_t0 = np.argmax(avgwf)

                    # Shift waveform to match t0 between events
                    shift = avgwf_t0 - 60
                    r1c_shift = np.zeros((self.n_pixels, self.n_samples))
                    if shift >= 0:
                        r1c_shift[:, :r1c[:, shift:].shape[1]] = r1c[:, shift:]
                    else:
                        r1c_shift[:, r1c[:, shift:].shape[1]:] = r1c[:, :shift]

                    # Get average pulse of event after shift
                    avgwf_shifted = np.mean(r1c_shift[connected], 0)

                    # Get pixel wf
                    wf = r1c[self.wf_poi]
                    wf_shifted = r1c_shift[self.wf_poi]

                    a_row[c][ev] = row
                    a_area[c][ev] = np.sum(r1c_shift[:, ws:we], 1)
                    a_height[c][ev] = np.max(r1c_shift, 1)
                    a_peakpos[c][ev] = np.argmax(r1c_shift, 1)
                    a_height_at_t0[c][ev] = r1c_shift[:, match_t0]
                    a_height_in_window[c][ev] = np.max(r1c_shift[:, ws:we], 1)
                    a_peakpos_in_window[c][ev] = np.argmax(r1c_shift[:, ws:we], 1) + ws

                    # Fill waveform dataframe
                    dfwf_list.append(dict(
                        cleaner=c,
                        event=ev,
                        row=row[self.wf_poi],
                        avgwf=avgwf,
                        wf=wf,
                        avgwf_shifted=avgwf_shifted,
                        wf_shifted=wf_shifted
                    ))

            # Fill dataframe
            for c in self.cleaners.keys():
                a_c = np.full((n_events, self.n_pixels), c)
                d_cleaner = dict(
                    cleaner=a_c.ravel(),
                    event=a_event.ravel(),
                    pixel=a_pixel.ravel(),
                    row=a_row[c].ravel(),
                    area=a_area[c].ravel(),
                    height=a_height[c].ravel(),
                    peakpos=a_peakpos[c].ravel(),
                    height_at_t0=a_height_at_t0[c].ravel(),
                    height_in_window=a_height_in_window[c].ravel(),
                    peakpos_in_window=a_peakpos_in_window[c].ravel()
                )
                df_cleaner = pd.DataFrame(d_cleaner)
                df_list.append(df_cleaner)

            df = pd.concat(df_list)
            dfwf = pd.DataFrame(dfwf_list)

            self.df = df
            self.dfwf = dfwf

            store = pd.HDFStore(join(self.output_dir, "data.h5"))
            store['df'] = df
            store['dfwf'] = dfwf

        store = pd.HDFStore(join(self.output_dir, "data.h5"))
        self.df = store['df']
        self.dfwf = store['dfwf']

    def finish(self):
        df = self.df
        dfwf = self.dfwf

        # Row 0 has problems in ped subtraction
        df = df.loc[df['row'] != 0]
        dfwf = dfwf.loc[dfwf['row'] != 0]

        # Cuts
        # b = ((df['cleaner'] == 'CleanerGaussConvolve') &
        #      (df['peakpos'] > 10) &
        #      (df['peakpos'] < 118))
        # df_c = df.loc[b]
        # df_c['cleaner'] = 'GaussConvolveWfEdgeExcluded'
        # df = pd.concat([df, df_c])

        cleaners = np.unique(df['cleaner'])

        for c in cleaners:
            b = (df['cleaner'] == c)
            df_c = df.loc[b]
            df.loc[b, 'area'] /= df_c['area'].mean()
            df.loc[b, 'height'] /= df_c['height'].mean()
            df.loc[b, 'height_at_t0'] /= df_c['height_at_t0'].mean()
            df.loc[b, 'height_in_window'] /= df_c['height_at_t0'].mean()

        df.loc[(df['pixel'] == 35) & (df['cleaner'] == 'CleanerCrossCorrelate'), 'height_at_t0']

        output_dir = join(self.output_dir, "p{}".format(self.wf_poi))
        if not exists(output_dir):
            self.log.info("Creating directory: {}".format(output_dir))
            makedirs(output_dir)

        # f_wf_pix = plt.figure(figsize=(14, 10))
        # ax = f_wf_pix.add_subplot(1, 1, 1)
        # dfwf_ev = dfwf.loc[dfwf['event'] == self.eoi]
        # for c in cleaners:
        #     wf = dfwf_ev.loc[dfwf_ev['cleaner'] == c, 'wf'].values[0]
        #     ax.plot(wf, label=c)
        # ax.set_title("Waveform (Event {}, Pixel {})".format(self.eoi, self.wf_poi))
        # ax.set_xlabel("Time (ns)")
        # ax.set_ylabel("Amplitude (ADC pedsub)")
        # ax.xaxis.set_minor_locator(MultipleLocator(1))
        # ax.legend(loc=1)
        # output_path = join(output_dir, "wf")
        # f_wf_pix.savefig(output_path, bbox_inches='tight')
        # self.log.info("Figure saved to: {}".format(output_path))
        #
        # f_wfshift_pix = plt.figure(figsize=(14, 10))
        # ax = f_wfshift_pix.add_subplot(1, 1, 1)
        # dfwf_ev = dfwf.loc[dfwf['event'] == self.eoi]
        # for c in cleaners:
        #     wf = dfwf_ev.loc[dfwf_ev['cleaner'] == c, 'wf_shifted'].values[0]
        #     ax.plot(wf, label=c)
        # plt.axvline(x=60, color="green")
        # plt.axvline(x=self.ws, color="red")
        # plt.axvline(x=self.we, color="red")
        # ax.set_title("Waveform Shifted (Event {}, Pixel {})".format(self.eoi, self.wf_poi))
        # ax.set_xlabel("Time (ns)")
        # ax.set_ylabel("Amplitude (ADC pedsub)")
        # ax.xaxis.set_minor_locator(MultipleLocator(1))
        # ax.legend(loc=1)
        # output_path = join(output_dir, "wf_shifted")
        # f_wfshift_pix.savefig(output_path, bbox_inches='tight')
        # self.log.info("Figure saved to: {}".format(output_path))
        #
        # f_avgwf_pix = plt.figure(figsize=(14, 10))
        # ax = f_avgwf_pix.add_subplot(1, 1, 1)
        # dfwf_ev = dfwf.loc[dfwf['event'] == self.eoi]
        # for c in cleaners:
        #     wf = dfwf_ev.loc[dfwf_ev['cleaner'] == c, 'avgwf'].values[0]
        #     ax.plot(wf, label=c)
        # ax.set_title("Average Waveform (Event {}, Pixel {})".format(self.eoi, self.wf_poi))
        # ax.set_xlabel("Time (ns)")
        # ax.set_ylabel("Amplitude (ADC pedsub)")
        # ax.xaxis.set_minor_locator(MultipleLocator(1))
        # ax.legend(loc=1)
        # output_path = join(output_dir, "avgwf")
        # f_avgwf_pix.savefig(output_path, bbox_inches='tight')
        # self.log.info("Figure saved to: {}".format(output_path))
        #
        # f_avgwfshift_pix = plt.figure(figsize=(14, 10))
        # ax = f_avgwfshift_pix.add_subplot(1, 1, 1)
        # dfwf_ev = dfwf.loc[dfwf['event'] == self.eoi]
        # for c in cleaners:
        #     wf = dfwf_ev.loc[dfwf_ev['cleaner'] == c, 'avgwf_shifted'].values[0]
        #     ax.plot(wf, label=c)
        # plt.axvline(x=60, color="green")
        # plt.axvline(x=self.ws, color="red")
        # plt.axvline(x=self.we, color="red")
        # ax.set_title("Average Waveform Shifted (Event {}, Pixel {})".format(self.eoi, self.wf_poi))
        # ax.set_xlabel("Time (ns)")
        # ax.set_ylabel("Amplitude (ADC pedsub)")
        # ax.xaxis.set_minor_locator(MultipleLocator(1))
        # ax.legend(loc=1)
        # output_path = join(output_dir, "avgwf_shifted")
        # f_avgwfshift_pix.savefig(output_path, bbox_inches='tight')
        # self.log.info("Figure saved to: {}".format(output_path))

        for pix in self.poi:
            df_pix = df.loc[df['pixel'] == pix]
            output_dir = join(self.output_dir, "p{}".format(pix))
            if not exists(output_dir):
                self.log.info("Creating directory: {}".format(output_dir))
                makedirs(output_dir)

            # f_area_spectrum = plt.figure(figsize=(14, 10))
            # ax = f_area_spectrum.add_subplot(1, 1, 1)
            # range_ = [-5, 15]
            # bins = 140
            # increment = (range_[1] - range_[0]) / bins
            # for c in cleaners:
            #     df_c = df_pix.loc[df_pix['cleaner']==c]
            #     v = df_c['area'].values
            #     ax.hist(v, bins=bins, range=range_, label=c, histtype='step')
            # ax.set_title("Area Spectrum (Pixel {})".format(pix))
            # ax.set_xlabel("Area")
            # ax.set_ylabel("N")
            # ax.xaxis.set_minor_locator(MultipleLocator(increment*2))
            # ax.xaxis.set_major_locator(MultipleLocator(increment*10))
            # ax.xaxis.grid(b=True, which='minor', alpha=0.5)
            # ax.xaxis.grid(b=True, which='major', alpha=0.8)
            # ax.legend(loc=1)
            # output_path = join(output_dir, "area_spectrum")
            # f_area_spectrum.savefig(output_path, bbox_inches='tight')
            # self.log.info("Figure saved to: {}".format(output_path))
            #
            # f_height_spectrum = plt.figure(figsize=(14, 10))
            # ax = f_height_spectrum.add_subplot(1, 1, 1)
            # range_ = [-1, 5]
            # bins = 110
            # increment = (range_[1] - range_[0]) / bins
            # for c in cleaners:
            #     df_c = df_pix.loc[df_pix['cleaner']==c]
            #     v = df_c['height'].values
            #     ax.hist(v, bins=bins, range=range_, label=c, histtype='step')
            # ax.set_title("Height Spectrum (Pixel {})".format(pix))
            # ax.set_xlabel("Height")
            # ax.set_ylabel("N")
            # ax.xaxis.set_minor_locator(MultipleLocator(increment*2))
            # ax.xaxis.set_major_locator(MultipleLocator(increment*10))
            # ax.xaxis.grid(b=True, which='minor', alpha=0.5)
            # ax.xaxis.grid(b=True, which='major', alpha=0.8)
            # ax.legend(loc=1)
            # output_path = join(output_dir, "height_spectrum")
            # f_height_spectrum.savefig(output_path, bbox_inches='tight')
            # self.log.info("Figure saved to: {}".format(output_path))
            #
            # f_peakpos_spectrum = plt.figure(figsize=(14, 10))
            # ax = f_peakpos_spectrum.add_subplot(1, 1, 1)
            # range_ = [0, self.n_samples]
            # bins = self.n_samples
            # increment = (range_[1] - range_[0]) / bins
            # for c in cleaners:
            #     df_c = df_pix.loc[df_pix['cleaner']==c]
            #     v = df_c['peakpos'].values
            #     ax.hist(v, bins=bins, range=range_, label=c, histtype='step')
            # ax.set_title("Peakpos Spectrum (Pixel {})".format(pix))
            # ax.set_xlabel("Peakpos")
            # ax.set_ylabel("N")
            # ax.xaxis.set_minor_locator(MultipleLocator(increment*2))
            # ax.xaxis.set_major_locator(MultipleLocator(increment*10))
            # ax.xaxis.grid(b=True, which='minor', alpha=0.5)
            # ax.xaxis.grid(b=True, which='major', alpha=0.8)
            # ax.legend(loc=1)
            # output_path = join(output_dir, "peakpos_spectrum")
            # f_peakpos_spectrum.savefig(output_path, bbox_inches='tight')
            # self.log.info("Figure saved to: {}".format(output_path))
            #
            # f_heightatt0_spectrum = plt.figure(figsize=(14, 10))
            # ax = f_heightatt0_spectrum.add_subplot(1, 1, 1)
            # range_ = [-5, 15]
            # bins = 110
            # increment = (range_[1] - range_[0]) / bins
            # for c in cleaners:
            #     df_c = df_pix.loc[df_pix['cleaner']==c]
            #     v = df_c['height_at_t0'].values
            #     ax.hist(v, bins=bins, range=range_, label=c, histtype='step')
            # ax.set_title("Height At T0 Spectrum (Pixel {})".format(pix))
            # ax.set_xlabel("Height At T0")
            # ax.set_ylabel("N")
            # ax.xaxis.set_minor_locator(MultipleLocator(increment*2))
            # ax.xaxis.set_major_locator(MultipleLocator(increment*10))
            # ax.xaxis.grid(b=True, which='minor', alpha=0.5)
            # ax.xaxis.grid(b=True, which='major', alpha=0.8)
            # ax.legend(loc=1)
            # output_path = join(output_dir, "heightatt0_spectrum")
            # f_heightatt0_spectrum.savefig(output_path, bbox_inches='tight')
            # self.log.info("Figure saved to: {}".format(output_path))

            f_heightatt0_fit = plt.figure(figsize=(14, 10))
            ax = plt.subplot2grid((1,3), (0,0), colspan=2)#f_heightatt0_fit.add_subplot(1, 2, 1)
            axt = plt.subplot2grid((1,3), (0,2))#f_heightatt0_fit.add_subplot(1, 2, 2)
            range_ = [-5, 4]
            bins = 110
            increment = (range_[1] - range_[0]) / bins
            c = 'CleanerCrossCorrelate'
            df_c = df_pix.loc[df_pix['cleaner'] == c]
            v = df_c['height_at_t0'].values
            self.fitter.range = range_
            self.fitter.nbins = bins
            self.fitter.apply(v)
            h = self.fitter.hist
            e = self.fitter.edges
            b = self.fitter.between
            fitx = self.fitter.fit_x
            fit = self.fitter.fit
            coeff = self.fitter.coeff
            coeff_l = self.fitter.coeff_list
            ax.hist(b, bins=e, weights=h, histtype='step')
            ax.plot(fitx, fit, label="Fit")
            for sf in self.fitter.subfit_labels:
                arr = self.fitter.subfits[sf]
                ax.plot(fitx, arr, label=sf)
            ax.set_title("Height At T0 Fit (Pixel {}, Cleaner {})".format(pix, c))
            ax.set_xlabel("Height At T0")
            ax.set_ylabel("N")
            ax.xaxis.set_minor_locator(MultipleLocator(increment*2))
            ax.xaxis.set_major_locator(MultipleLocator(increment*10))
            ax.xaxis.grid(b=True, which='minor', alpha=0.5)
            ax.xaxis.grid(b=True, which='major', alpha=0.8)
            ax.legend(loc=1)
            axt.axis('off')
            table_data = [['%.3f' % coeff[i]] for i in coeff_l]
            table_row = coeff_l
            table = axt.table(cellText=table_data, rowLabels=table_row, loc='center')
            table.scale(1, 2)
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            output_path = join(output_dir, "heightatt0_fit")
            f_heightatt0_fit.savefig(output_path, bbox_inches='tight')
            self.log.info("Figure saved to: {}".format(output_path))

            # f_heightiw_spectrum = plt.figure(figsize=(14, 10))
            # ax = f_heightiw_spectrum.add_subplot(1, 1, 1)
            # range_ = [-5, 15]
            # bins = 110
            # increment = (range_[1] - range_[0]) / bins
            # for c in cleaners:
            #     df_c = df_pix.loc[df_pix['cleaner']==c]
            #     v = df_c['height_in_window'].values
            #     ax.hist(v, bins=bins, range=range_, label=c, histtype='step')
            # ax.set_title("Height In Window Spectrum (Pixel {})".format(pix))
            # ax.set_xlabel("Height In Window")
            # ax.set_ylabel("N")
            # ax.xaxis.set_minor_locator(MultipleLocator(increment*2))
            # ax.xaxis.set_major_locator(MultipleLocator(increment*10))
            # ax.xaxis.grid(b=True, which='minor', alpha=0.5)
            # ax.xaxis.grid(b=True, which='major', alpha=0.8)
            # ax.legend(loc=1)
            # output_path = join(output_dir, "heightiw_spectrum")
            # f_heightiw_spectrum.savefig(output_path, bbox_inches='tight')
            # self.log.info("Figure saved to: {}".format(output_path))
            #
            # f_peakposiw_spectrum = plt.figure(figsize=(14, 10))
            # ax = f_peakposiw_spectrum.add_subplot(1, 1, 1)
            # range_ = [0, self.n_samples]
            # bins = self.n_samples
            # increment = (range_[1] - range_[0]) / bins
            # for c in cleaners:
            #     df_c = df_pix.loc[df_pix['cleaner']==c]
            #     v = df_c['peakpos_in_window'].values
            #     ax.hist(v, bins=bins, range=range_, label=c, histtype='step')
            # ax.set_title("Peakpos In Window Spectrum (Pixel {})".format(pix))
            # ax.set_xlabel("Peakpos In Window")
            # ax.set_ylabel("N")
            # ax.xaxis.set_minor_locator(MultipleLocator(increment*2))
            # ax.xaxis.set_major_locator(MultipleLocator(increment*10))
            # ax.xaxis.grid(b=True, which='minor', alpha=0.5)
            # ax.xaxis.grid(b=True, which='major', alpha=0.8)
            # ax.legend(loc=1)
            # output_path = join(output_dir, "peakposiw_spectrum")
            # f_peakposiw_spectrum.savefig(output_path, bbox_inches='tight')
            # self.log.info("Figure saved to: {}".format(output_path))


if __name__ == '__main__':
    exe = SiPMSPETesting()
    exe.run()
