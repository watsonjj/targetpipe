from astropy import log
import numpy as np
from math import ceil
from IPython import embed


class ToyWaveformsCHECM:
    """
    Produce fake waveforms that resemble those produced by CHEC-M

    Attributes
    ----------
    n_events : int
        number of events to be simulated. Can be set with
        `init_all_cells_filled`
    n_pix : int
        number of camera pixels
    n_cell : int
        number of cells per pixel
    n_samples : int
        number of samples per pixel
    ped_mean : int
        mean of the pedestal. Unit=ADC
    ped_rms : int
        intrinsic rms of the pedestal (without cells/pixel affects). Unit=ADC
    pulse_min : int
        minimum pulse height. Unit=ADC
    pulse_width_mean : int
        mean of the distribution of pulse width rms. Unit=ns
    pulse_width_rms : int
        rms of the distrubution of pulse width rms. Unis=ns
    pulse_t_mean : int
        mean of the distribution of pulse peaktime mean. Unit=ns
    pulse_t_rms : int
        rms of the distribution of pulse peaktime mean. Unit=ns

    """
    def __init__(self, n_events):
        self.n_events = n_events
        self.n_pix = 2048
        self.n_cell = 16384
        self.n_samples = 128

        self.ped_mean = 400
        self.ped_rms = 2

        self.pulse_min = 0
        self.pulse_max = 100

        self.pulse_width_mean = 6
        self.pulse_width_rms = 1

        self.pulse_t_mean = 40
        self.pulse_t_rms = 5

        self.__cell_random = True

    def init_all_cells_filled(self, fill_factor):
        """
        Init n_events to be the number of events required to fill all cells,
        depending on the n_samples currently set.

        First_cell_ids are no longer set randomly, and instead are filled in
        sequence to minimise n_events required to fill each cell at least
        `fill_factor` times.

        Parameters
        ----------
        fill_factor : int
            Fill all cells at least `fill_factor` times.

        """
        self.n_events = int(ceil(fill_factor * self.n_cell / self.n_samples))
        log.info("n_events = {}".format(self.n_events))

    def __reinit(self):
        self._n_ts = self.n_events * self.n_samples  # n_total_samples
        self._p_shift = np.random.randint(-200, 200, size=self.n_pix)
        self._c_shift = np.random.randint(-50, 50, size=(self.n_pix,
                                                         self.n_cell))
        max_repeats = int(ceil(self._n_ts / self.n_cell))
        # noinspection PyTypeChecker
        cell_shift_tiled = np.tile(self._c_shift, max_repeats)
        self._c_shift = np.delete(cell_shift_tiled, np.s_[self._n_ts:], 1)

        self.cells = np.arange(0, self._n_ts, dtype=np.uint16)
        self.cells %= self.n_cell
        embed()

    def get_base_wf(self):
        """
        Obtain toy-waveforms that resemble pedestal data (no pulses)

        Returns
        -------
        ndarray
            Numpy array containing waveforms and cell ids

        """
        self.__reinit()
        log.info("Producing base toy waveforms")

        wf_c_shape = (self.n_pix, 2, self._n_ts)
        wf_c = np.empty(wf_c_shape, dtype=np.uint16)
        wf_c_c = wf_c[:, 0, :]
        wf_c_w = wf_c[:, 1, :]

        cells = np.arange(0, self._n_ts, dtype=np.uint16)
        cells[cells >= self.n_cell] %= self.n_cell
        wf_c_c[:] = cells[:]

        ped = (self.ped_mean, self.ped_rms)
        wf_c_w[:] = np.random.normal(*ped, size=wf_c_w.shape)

        wf_c_w[:] = (wf_c_w[:] + self._c_shift).astype(np.uint16)
        wf_c_w[:] = (wf_c_w[:] + self._p_shift[:, None]).astype(np.uint16)

        wf_c = np.array(np.split(wf_c, self.n_events, axis=2))
        np.random.shuffle(wf_c)

        return wf_c

    def get_pulse_wf(self):
        """
        Obtain toy-waveforms with pulses

        Returns
        -------
        ndarray
            Numpy array containing waveforms and cell ids

        """
        pulse_waveforms = self.get_base_wf()
        log.info("Producing toy waveforms with pulses")

        pulse_height = np.random.randint(self.pulse_min, self.pulse_max,
                                         size=(self.n_events, self.n_pix))
        pulse_mean = np.random.normal(self.pulse_t_mean,
                                      self.pulse_t_rms,
                                      size=(self.n_events, self.n_pix))
        pulse_rms = np.random.normal(self.pulse_width_mean,
                                     self.pulse_width_rms,
                                     size=(self.n_events, self.n_pix))

        def pulse_shape(h, mu, sig):
            x = np.arange(self.n_samples)
            return h * np.exp((-1/2)*np.power((x - mu)/sig, 2.))
        gaussians = np.zeros((self.n_events, self.n_pix, self.n_samples),
                             dtype=np.uint16)
        for ev in range(self.n_events):
            for p in range(self.n_pix):
                gaussians[ev, p, :] = pulse_shape(pulse_height[ev, p],
                                                  pulse_mean[ev, p],
                                                  pulse_rms[ev, p])

        pulse_waveforms[:, :, 1, :] += gaussians

        return pulse_waveforms
