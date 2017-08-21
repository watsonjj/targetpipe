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

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from ctapipe.image.waveform_cleaning import WaveformCleanerFactory
from ctapipe.io.eventfilereader import EventFileReaderFactory


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

        self.n_pixels = None
        self.n_samples = None

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

        first_event = self.reader.get_event(0)
        self.n_pixels = first_event.inst.num_pixels[0]
        self.n_samples = first_event.r0.tel[0].num_samples

    def start(self):
        # connected = [40, 41, 42, 43]
        connected = list(range(self.n_pixels))
        ignore_pixels = [25, 26, 18, 19, 13, 14, 5, 6, 49, 37]
        connected = [i for i in connected if i not in ignore_pixels]
        self.n_pixels = len(connected)
        pix = 35
        pix_index = connected.index(pix)
        shift = 4
        width = 8
        match_t0 = 60
        within = 3

        window_start = match_t0 - shift
        if window_start < 0:
            window_start = 0
        window_end = window_start + width
        if window_end > self.n_samples - 1:
            window_end = self.n_samples - 1

        n_events = self.reader.num_events
        mask = np.zeros(n_events, dtype=np.bool)
        area = np.zeros(n_events)
        height_t0 = np.zeros(n_events)
        height_peak = np.zeros(n_events)
        peakpos_iw = np.zeros(n_events)
        peakpos = np.zeros(n_events)
        t0 = np.zeros(n_events)
        t0_chk = np.zeros(n_events)
        waveform_pix = np.zeros((n_events, self.n_samples))
        waveform_avg = np.zeros((n_events, self.n_samples))
        fwhm = np.zeros(n_events)
        rise_time = np.zeros(n_events)

        ind = np.indices((self.n_pixels, self.n_samples))[1]
        r_ind = ind[:, ::-1]

        cut_row = 0
        cut_t0startend = 0
        cut_avg = 0

        source = self.reader.read()
        desc = "Looping through file"
        for event in tqdm(source, desc=desc, total=n_events):
            ev = event.count

            # Skip first row due to problem in pedestal subtraction
            row = event.r0.tel[0].row[0]
            if row == 0:
                cut_row += 1
                mask[ev] = True
                continue

            # Calibrate
            self.r1.calibrate(event)
            r1 = event.r1.tel[0].pe_samples[0][connected]

            # # Subtract baseline
            # baseline_ev = np.median(r1[:, :40], 1)
            # r1 = r1 - baseline_ev[:, None]

            # Waveform cleaning
            kernel = general_gaussian(5, p=1.0, sig=1)
            smooth_flat = np.convolve(r1.ravel(), kernel, "same")
            smoothed = np.reshape(smooth_flat, r1.shape)
            samples_std = np.std(r1, axis=1)
            smooth_baseline_std = np.std(smoothed, axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                smoothed *= (samples_std / smooth_baseline_std)[:, None]
                smoothed[~np.isfinite(smoothed)] = 0
            r1 = smoothed

            # Get average pulse of event
            avg = r1.mean(0)

            # if avg[20:40].max() < 2:
            #     mask[ev] = True
            #     continue
            #
            # r = event.r0.tel[0].row[0]
            # c = event.r0.tel[0].column[0]
            # bp = event.r0.tel[0].blockphase[0]
            # blk = r + c * 8
            # fci = event.r0.tel[0].first_cell_ids[0]
            # print(ev, fci, blk, r, c, bp)

            # # Skip events with a low average maximum
            # if avg.max() < 4:
            #     cut_avg += 1
            #     mask[ev] = True
            #     continue

            # Get t0 of average pulse for event
            # TODO: fit average pulse
            t0_ev = avg.argmax()

            # Skip events with t0 close to start or end of waveform
            if t0_ev < 10 or t0_ev > 118:
                cut_t0startend += 1
                mask[ev] = True
                continue

            # # Skip events outside acceptable t0 range
            # if t0_ev < 70 or t0_ev > 75:
            #     mask[ev] = True
            #     continue

            # Shift waveform to match t0 between events
            t0_ev_shift = t0_ev - 60
            r1_shift = np.zeros((self.n_pixels, self.n_samples))
            if t0_ev_shift >= 0:
                r1_shift[:, :r1[:, t0_ev_shift:].shape[1]] = r1[:, t0_ev_shift:]
            else:
                r1_shift[:, r1[:, t0_ev_shift:].shape[1]:] = r1[:, :t0_ev_shift]

            # Check t0 matching
            avg_chk = r1_shift.mean(0)
            t0_chk_ev = avg_chk.argmax()

            r1_pix = r1_shift[pix_index]
            area_ev = r1_pix[window_start:window_end].sum()
            height_t0_ev = r1_pix[match_t0]
            height_peak_ev = r1_pix[window_start:window_end].max()
            peakpos_iw_ev = r1_pix[window_start:window_end].argmax() + window_start
            peakpos_ev = r1_pix.argmax()

            # Skip events with peakpos outside acceptable range
            # if abs(peakpos_ev - match_t0) > within:
            #     mask[ev] = True
            #     continue

            ind_pix = ind[pix_index]
            r_ind_pix = r_ind[pix_index]
            max_ = np.max(r1_pix)
            reversed_ = r1_pix[::-1]
            mask_before = np.ma.masked_less(ind_pix, peakpos_ev).mask
            mask_after = np.ma.masked_greater(r_ind_pix, peakpos_ev).mask
            masked_bef = np.ma.masked_array(r1_pix, mask_before)
            masked_aft = np.ma.masked_array(reversed_, mask_after)
            half_max = max_/2
            d_l = np.diff(np.sign(half_max - masked_aft))
            d_r = np.diff(np.sign(half_max - masked_bef))
            t_l = r_ind_pix[np.argmax(d_l)+1]
            t_r = ind_pix[np.argmax(d_r)+1]
            fwhm_ev = t_r - t_l
            _10percent = 0.1 * max_
            _90percent = 0.9 * max_
            d10 = np.diff(np.sign(_10percent - masked_aft))
            d90 = np.diff(np.sign(_90percent - masked_aft))
            t10 = r_ind_pix[np.argmax(d10)+1]
            t90 = r_ind_pix[np.argmax(d90)+1]
            rise_time_ev = t90 - t10

            # plt.plot(r1_pix)
            # plt.axvline(t_r, color='red')
            # plt.axvline(t_l, color='red')
            # plt.axvline(t10, color='green')
            # plt.axvline(t90, color='green')
            # plt.axhline(half_max, color='red')
            # plt.axhline(_10percent, color='green')
            # plt.axhline(_90percent, color='green')
            # plt.show()

            area[ev] = area_ev
            height_t0[ev] = height_t0_ev
            height_peak[ev] = height_peak_ev
            peakpos_iw[ev] = peakpos_iw_ev
            peakpos[ev] = peakpos_ev
            t0[ev] = t0_ev
            t0_chk[ev] = t0_chk_ev
            waveform_pix[ev] = r1_pix
            waveform_avg[ev] = avg
            fwhm[ev] = fwhm_ev
            rise_time[ev] = rise_time_ev

        def remove_events(array):
            array = np.ma.masked_array(array, mask=mask).compressed()
            return array

        def remove_events_samples(array):
            mask_samples = np.zeros((n_events, self.n_samples), dtype=np.bool)
            mask_samples = np.ma.mask_or(mask_samples, mask[:, None])
            array = np.ma.masked_array(array, mask=mask_samples).compressed()
            array = array.reshape((array.size // self.n_samples, self.n_samples))
            return array

        area = remove_events(area)
        height_t0 = remove_events(height_t0)
        height_peak = remove_events(height_peak)
        peakpos_iw = remove_events(peakpos_iw)
        peakpos = remove_events(peakpos)
        t0 = remove_events(t0)
        t0_chk = remove_events(t0_chk)
        fwhm = remove_events(fwhm)
        fwhm = np.ma.masked_where(fwhm < 0, fwhm).compressed()
        rise_time = remove_events(rise_time)
        waveform_pix = remove_events_samples(waveform_pix)
        waveform_avg = remove_events_samples(waveform_avg)

        self.log.info("Number of events removed by row cut = {}".format(cut_row))
        self.log.info("Number of events removed by t0 close to start or "
                      "end of waveform = {}".format(cut_t0startend))
        self.log.info("Number of events removed by a low avg "
                      "max = {}".format(cut_avg))
        self.log.info("Number of events after cuts = {}".format(area.size))

        output_dir = join(self.reader.output_directory, "sipm_spe_testing")
        output_dir = join(output_dir, "ch{}".format(pix))
        if not exists(output_dir):
            self.log.info("Creating directory: {}".format(output_dir))
            makedirs(output_dir)

        f_area_spectrum = plt.figure(figsize=(14, 10))
        ax = f_area_spectrum.add_subplot(1, 1, 1)
        range_ = [-80, 340]
        bins = 140
        increment = (range_[1] - range_[0]) / bins
        ax.hist(area, bins=bins, range=range_)
        ax.set_title("Area Spectrum (Channel {})".format(pix))
        ax.set_xlabel("Area")
        ax.set_ylabel("N")
        ax.xaxis.set_minor_locator(MultipleLocator(increment*2))
        ax.xaxis.set_major_locator(MultipleLocator(increment*10))
        ax.xaxis.grid(b=True, which='minor', alpha=0.5)
        ax.xaxis.grid(b=True, which='major', alpha=0.8)
        output_path = join(output_dir, "area_spectrum")
        f_area_spectrum.savefig(output_path, bbox_inches='tight')
        self.log.info("Figure saved to: {}".format(output_path))

        f_height_t0_spectrum = plt.figure(figsize=(14, 10))
        ax = f_height_t0_spectrum.add_subplot(1, 1, 1)
        range_ = [-10, 45]
        bins = 110
        increment = (range_[1] - range_[0]) / bins
        ax.hist(height_t0, bins=bins, range=range_)
        ax.set_title("Height t0 Spectrum (Channel {})".format(pix))
        ax.set_xlabel("Height (@t0)")
        ax.set_ylabel("N")
        ax.xaxis.set_minor_locator(MultipleLocator(increment*2))
        ax.xaxis.set_major_locator(MultipleLocator(increment*10))
        ax.xaxis.grid(b=True, which='minor', alpha=0.5)
        ax.xaxis.grid(b=True, which='major', alpha=0.8)
        ax.xaxis.set_minor_locator(MultipleLocator(increment))
        output_path = join(output_dir, "height_t0_spectrum")
        f_height_t0_spectrum.savefig(output_path, bbox_inches='tight')
        self.log.info("Figure saved to: {}".format(output_path))

        f_height_peak_spectrum = plt.figure(figsize=(14, 10))
        ax = f_height_peak_spectrum.add_subplot(1, 1, 1)
        range_ = [-5, 50]
        bins = 110
        increment = (range_[1] - range_[0]) / bins
        ax.hist(height_peak, bins=bins, range=range_)
        ax.set_title("Height Peak Spectrum (Channel {})".format(pix))
        ax.set_xlabel("Height (@Peak)")
        ax.set_ylabel("N")
        ax.xaxis.set_minor_locator(MultipleLocator(increment*2))
        ax.xaxis.set_major_locator(MultipleLocator(increment*10))
        ax.xaxis.grid(b=True, which='minor', alpha=0.5)
        ax.xaxis.grid(b=True, which='major', alpha=0.8)
        ax.xaxis.set_minor_locator(MultipleLocator(increment))
        output_path = join(output_dir, "height_peak_spectrum")
        f_height_peak_spectrum.savefig(output_path, bbox_inches='tight')
        self.log.info("Figure saved to: {}".format(output_path))

        f_peakpos_iw_hist = plt.figure(figsize=(14, 10))
        ax = f_peakpos_iw_hist.add_subplot(1, 1, 1)
        min_ = peakpos_iw.min()
        max_ = peakpos_iw.max()
        nbins = int(max_ - min_)
        n, e, _ = ax.hist(peakpos_iw, bins=nbins, range=[min_, max_])
        ax.text(0.99, 0.99, 'Max = {}'.format(e[n.argmax()]),
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes, color='green', fontsize=15)
        ax.set_title("peakpos Distribution inside window(Channel {})".format(pix))
        ax.set_xlabel("peakpos")
        ax.set_ylabel("N")
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        output_path = join(output_dir, "peakpos_iw_hist")
        f_peakpos_iw_hist.savefig(output_path, bbox_inches='tight')
        self.log.info("Figure saved to: {}".format(output_path))

        f_peakpos_hist = plt.figure(figsize=(14, 10))
        ax = f_peakpos_hist.add_subplot(1, 1, 1)
        min_ = peakpos.min()
        max_ = peakpos.max()
        nbins = int(max_ - min_)
        n, e, _ = ax.hist(peakpos, bins=nbins, range=[min_, max_])
        ax.text(0.99, 0.99, 'Max = {}'.format(e[n.argmax()]),
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes, color='green', fontsize=15)
        ax.set_title("peakpos Distribution from whole waveform(Channel {})".format(pix))
        ax.set_xlabel("peakpos")
        ax.set_ylabel("N")
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        output_path = join(output_dir, "peakpos_hist")
        f_peakpos_hist.savefig(output_path, bbox_inches='tight')
        self.log.info("Figure saved to: {}".format(output_path))

        f_t0_hist = plt.figure(figsize=(14, 10))
        ax = f_t0_hist.add_subplot(1, 1, 1)
        min_ = t0.min()
        max_ = t0.max()
        nbins = int(max_ - min_)
        n, e, _ = ax.hist(t0, bins=nbins, range=[min_, max_])
        ax.text(0.99, 0.99, 'Max = {}'.format(e[n.argmax()]),
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes, color='green', fontsize=15)
        ax.set_title("t0 Distribution")
        ax.set_xlabel("t0")
        ax.set_ylabel("N")
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        output_path = join(output_dir, "t0_hist")
        f_t0_hist.savefig(output_path, bbox_inches='tight')
        self.log.info("Figure saved to: {}".format(output_path))

        f_t0_chk_hist = plt.figure(figsize=(14, 10))
        ax = f_t0_chk_hist.add_subplot(1, 1, 1)
        min_ = t0_chk.min()
        max_ = t0_chk.max()
        nbins = int(max_ - min_ + 1)
        n, e, _ = ax.hist(t0_chk, bins=nbins, range=[min_, max_])
        ax.text(0.99, 0.99, 'Max = {}'.format(e[n.argmax()]),
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes, color='green', fontsize=15)
        ax.set_title("t0 (matched) Distribution")
        ax.set_xlabel("t0")
        ax.set_ylabel("N")
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        output_path = join(output_dir, "t0_chk_hist")
        f_t0_chk_hist.savefig(output_path, bbox_inches='tight')
        self.log.info("Figure saved to: {}".format(output_path))

        f_fwhm_hist = plt.figure(figsize=(14, 10))
        ax = f_fwhm_hist.add_subplot(1, 1, 1)
        min_ = fwhm.min()
        max_ = fwhm.max()
        nbins = int(max_ - min_)
        n, e, _ = ax.hist(fwhm, bins=nbins, range=[min_, max_])
        ax.text(0.99, 0.99, 'Max = {}'.format(e[n.argmax()]),
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes, color='green', fontsize=15)
        ax.set_title("fwhm Distribution (Channel {})".format(pix))
        ax.set_xlabel("fwhm")
        ax.set_ylabel("N")
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        output_path = join(output_dir, "fwhm_hist")
        f_fwhm_hist.savefig(output_path, bbox_inches='tight')
        self.log.info("Figure saved to: {}".format(output_path))

        f_rise_time_hist = plt.figure(figsize=(14, 10))
        ax = f_rise_time_hist.add_subplot(1, 1, 1)
        min_ = rise_time.min()
        max_ = rise_time.max()
        nbins = int(max_ - min_)
        n, e, _ = ax.hist(rise_time, bins=nbins, range=[min_, max_])
        ax.text(0.99, 0.99, 'Max = {}'.format(e[n.argmax()]),
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes, color='green', fontsize=15)
        ax.set_title("rise_time Distribution (Channel {})".format(pix))
        ax.set_xlabel("rise_time")
        ax.set_ylabel("N")
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        output_path = join(output_dir, "rise_time_hist")
        f_rise_time_hist.savefig(output_path, bbox_inches='tight')
        self.log.info("Figure saved to: {}".format(output_path))

        f_waveforms_pix = plt.figure(figsize=(14, 10))
        ax = f_waveforms_pix.add_subplot(1, 1, 1)
        ax.plot(np.rollaxis(waveform_pix, 1))
        plt.axvline(x=60, color="green")
        plt.axvline(x=window_start, color="red")
        plt.axvline(x=window_end, color="red")
        ax.set_title("Waveforms (Channel {})".format(pix))
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude (ADC pedsub)")
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        output_path = join(output_dir, "waveforms_pix")
        f_waveforms_pix.savefig(output_path, bbox_inches='tight')
        self.log.info("Figure saved to: {}".format(output_path))

        f_waveforms_pix_hist = plt.figure(figsize=(14, 10))
        ax = f_waveforms_pix_hist.add_subplot(1, 1, 1)
        x = np.indices(waveform_pix.shape)[1].ravel()
        y = waveform_pix.ravel()
        hb = ax.hexbin(x, y, bins='log')
        cb = f_waveforms_pix_hist.colorbar(hb, ax=ax)
        cb.set_label('Counts (log)')
        plt.axvline(x=60, color="green")
        plt.axvline(x=window_start, color="red")
        plt.axvline(x=window_end, color="red")
        ax.set_title("Waveforms (Channel {})".format(pix))
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude (ADC pedsub)")
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        output_path = join(output_dir, "waveforms_pix_hist")
        f_waveforms_pix_hist.savefig(output_path, bbox_inches='tight')
        self.log.info("Figure saved to: {}".format(output_path))

        f_waveforms_pix_hist_spe = plt.figure(figsize=(14, 10))
        ax = f_waveforms_pix_hist_spe.add_subplot(1, 1, 1)
        x = np.indices(waveform_pix.shape)[1].ravel()
        y = waveform_pix.ravel()
        yrange_ = [-5, 10]
        ybins = 110
        xrange_ = [-0.5, 127.5]
        xbins = 128
        h2, ex, ey = np.histogram2d(x, y, bins=[xbins, ybins], range=[xrange_, yrange_])
        scale = LogNorm(vmin=160, vmax=277)
        p = plt.pcolormesh(ex, ey, h2.T, norm=scale)
        cb = f_waveforms_pix_hist_spe.colorbar(p, ax=ax, extend='max')
        cb.set_label('Counts (log)')
        plt.axvline(x=60, color="green")
        plt.axvline(x=window_start, color="red")
        plt.axvline(x=window_end, color="red")
        ax.set_title("Waveforms (Channel {})".format(pix))
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude (ADC pedsub)")
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        output_path = join(output_dir, "waveforms_pix_hist_spe")
        f_waveforms_pix_hist_spe.savefig(output_path, bbox_inches='tight')
        self.log.info("Figure saved to: {}".format(output_path))

        f_waveforms_avg = plt.figure(figsize=(14, 10))
        ax = f_waveforms_avg.add_subplot(1, 1, 1)
        ax.plot(np.rollaxis(waveform_avg, 1))
        ax.set_title("Average Waveforms")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude (ADC pedsub)")
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        output_path = join(output_dir, "waveforms_avg")
        f_waveforms_avg.savefig(output_path, bbox_inches='tight')
        self.log.info("Figure saved to: {}".format(output_path))

        output_path = join(output_dir, "area.npy")
        np.save(output_path, area)
        self.log.info("Numpy array saved to: {}".format(output_path))

    def finish(self):
        pass


if __name__ == '__main__':
    exe = SiPMSPETesting()
    exe.run()
