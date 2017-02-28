"""
Functions and classes used to generate rolling statistics, i.e. stastics that
are calculated and returned each time you supply a new value
"""

import numpy as np


class NumpyMeanStdDev:
    def __init__(self, shape):
        self.shape = shape
        self.n = np.zeros(shape)
        self.S = np.zeros(shape)
        self.m = np.zeros(shape)
        self.o = np.zeros(shape)

        self.mean = np.zeros(shape)
        self.variance = np.zeros(shape)
        self.stddev = np.zeros(shape)

    def send(self, x, s=np.s_[:]):
        self.n[s] += 1
        m_prev = self.m[s]
        self.m[s] += (x - self.m[s]) / self.n[s]
        self.S[s] += (x - self.m[s]) * (x - m_prev)
        self.mean[s] = self.m[s]
        s_r = self.n[s] > 1
        self.variance[s][s_r] = self.S[s][s_r] / (self.n[s][s_r] - 1)
        self.stddev[s] = np.sqrt(self.variance[s])


class PedestalMeanStdDev(NumpyMeanStdDev):
    def __init__(self, n_pix, n_cells):
        self.n_cells = n_cells
        super().__init__((n_pix, n_cells))

    def send_waveform(self, waveforms, first_cell_ids):
        """

        Parameters
        ----------
        waveforms : ndarray
            Numpy array of shape (n_pix, n_samples) containing waveforms
        first_cell_ids : ndarray
            Numpy array of shape (n_pix) containing first_cell_ids
        """
        n_pix = waveforms.shape[0]
        n_samples = waveforms.shape[1]
        fci = first_cell_ids

        first_cell_ids_changes = list(np.where(np.diff(fci) != 0)[0])
        first_cell_ids_changes.append(n_pix-1)
        start = 0
        for i in first_cell_ids_changes:
            index = i+1
            if fci[start] + n_samples > self.n_cells:
                initial = self.n_cells-fci[start]
                remain = n_samples - initial
                s = np.s_[start:index, fci[start]:self.n_cells]
                self.send(waveforms[start:index, 0:initial], s)
                s = np.s_[start:index, 0:remain]
                self.send(waveforms[start:index, initial:], s)
            else:
                s = np.s_[start:index, fci[start]:fci[start] + n_samples]
                self.send(waveforms[start:index], s)
            start = index
