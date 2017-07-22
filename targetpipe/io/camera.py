import numpy as np
from astropy import log
from os.path import dirname, realpath, join
from ctapipe.instrument.camera import _get_min_pixel_seperation
from ctapipe.instrument import CameraGeometry
from astropy import units as u
from target_io import TargetIOEventReader as TIOReader, \
    T_SAMPLES_PER_WAVEFORM_BLOCK as N_BLOCKSAMPLES


class Borg:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class Config(Borg):

    def __init__(self, camera_id=None):
        Borg.__init__(self)
        if not self.__dict__:
            self._id = None

            self.dir = dirname(realpath(__file__))

            self._cameraname = None

            self.n_pix = None
            self.optical_foclen = None
            self.pixel_id_path = None
            self.pixel_pos_path = None
            self.ref_pulse_path = None
            self.pixel_id = None
            self.pixel_pos = None
            self.refshape = None
            self.refstep = None
            self.time_slice = None
            self.dead_pixels = None

            self.n_rows = None
            self.n_columns = None
            self.n_blocksamples = None
            self.n_blocks = None
            self.n_cells = None
            self.skip_sample = None
            self.skip_end_sample = None
            self.skip_event = None
            self.skip_end_event = None

            self.options = dict(
                checm=self._case_checm,
                checm_single=self._case_checm_single,
                checs=self._case_checs,
                checs_single=self._case_checs_single
            )

            if not camera_id:
                self.id = 'checs'

        if camera_id:
            self.id = camera_id

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, val):
        if not self.id == val:
            self.switch_camera(val)

    def switch_camera(self, id):
        log.info("Loading camera config: {}".format(id))
        self._id = id
        try:
            self.options[id]()
        except KeyError:
            log.error("No camera with id: {}".format(id))
            raise

        self.pixel_id = np.load(self.pixel_id_path)[:self.n_pix]
        self.pixel_pos = np.load(self.pixel_pos_path)[:, :self.n_pix]
        ref_pulse_file = np.load(self.ref_pulse_path)
        self.refshape = ref_pulse_file['refshape']
        self.refstep = ref_pulse_file['refstep']
        self.time_slice = ref_pulse_file['time_slice']
        self.n_blocksamples = N_BLOCKSAMPLES
        self.n_blocks = self.n_rows * self.n_columns
        self.n_cells = self.n_rows * self.n_columns * self.n_blocksamples

    def switch_to_single_module(self):
        self.id = self._cameraname + '_single'

    def _case_checm(self):
        self._cameraname = 'checm'
        self.n_pix = 2048
        self.dead_pixels = [96, 276, 1906, 1910, 1916]
        self.optical_foclen = 2.283
        self.pixel_id_path = join(self.dir, 'checm_pixel_id.npy')
        self.pixel_pos_path = join(self.dir, 'checm_pixel_pos.npy')
        self.ref_pulse_path = join(self.dir, 'checm_reference_pulse.npz')
        self.n_rows = 8
        self.n_columns = 64
        self.skip_sample = 32
        self.skip_end_sample = 0
        self.skip_event = 2
        self.skip_end_event = 1

    def _case_checm_single(self):
        self._case_checm()
        self.n_pix = 64
        self.dead_pixels = []

    def _case_checs(self):
        self._cameraname = 'checs'
        self.n_pix = 2048
        self.dead_pixels = []
        self.optical_foclen = 2.283
        self.pixel_id_path = join(self.dir, 'checm_pixel_id.npy')  # TODO
        self.pixel_pos_path = join(self.dir, 'checm_pixel_pos.npy')  # TODO
        self.ref_pulse_path = join(self.dir, 'checm_reference_pulse.npz')  # TODO
        self.n_rows = 8
        self.n_columns = 16
        self.skip_sample = 0
        self.skip_end_sample = 0
        self.skip_event = 2
        self.skip_end_event = 1

    def _case_checs_single(self):
        self._case_checs()
        self.n_pix = 64
        self.dead_pixels = []
