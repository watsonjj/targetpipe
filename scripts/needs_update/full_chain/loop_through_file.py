from tqdm import tqdm
from traitlets import Dict, List
from matplotlib import pyplot as plt
import numpy as np
from IPython import embed

from ctapipe.calib.camera.calibrator import CameraCalibrator
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.core import Tool, Component
from ctapipe.image import tailcuts_clean
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from ctapipe.image.waveform_cleaning import WaveformCleanerFactory
from ctapipe.instrument import CameraGeometry
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.coordinates import CameraFrame, HorizonFrame, NominalFrame, TiltedGroundFrame
from astropy import units as u
from ctapipe.image.hillas import HillasParameterizationError, \
    hillas_parameters
from ctapipe.visualization import CameraDisplay



class Geometry(Component):
    def __init__(self, config, tool, event, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

        self.camera_geom_dict = {}
        self.nominal_geom_dict = {}

        self.inst = event.inst
        array_pointing = HorizonFrame(
            alt=event.mcheader.run_array_direction[1] * u.rad,
            az=event.mcheader.run_array_direction[0] * u.rad)
        self.nom_system = NominalFrame(array_direction=array_pointing,
                                       pointing_direction=array_pointing)

    def get_camera(self, tel_id):
        if not tel_id in self.camera_geom_dict:
            pos = self.inst.pixel_pos[tel_id]
            foclen = self.inst.optical_foclen[tel_id]
            geom = CameraGeometry.guess(*pos, foclen, apply_derotation=False)
            self.camera_geom_dict[tel_id] = geom
        return self.camera_geom_dict[tel_id]

    def get_nominal(self, tel_id):
        if not tel_id in self.nominal_geom_dict:
            camera_geom = self.get_camera(tel_id)
            pix_x, pix_y = self.inst.pixel_pos[tel_id]
            foclen = self.inst.optical_foclen[tel_id]
            camera_coord = CameraFrame(x=pix_x, y=pix_y, focal_length=foclen,
                                       rotation=-1 * camera_geom.cam_rotation)
            nom_coord = camera_coord.transform_to(self.nom_system)
            geom = CameraGeometry(camera_geom.cam_id, camera_geom.pix_id,
                                  nom_coord.x, nom_coord.y,
                                  None, camera_geom.pix_type,
                                  camera_geom.pix_rotation,
                                  camera_geom.cam_rotation,
                                  camera_geom.neighbors, False)
            self.nominal_geom_dict[tel_id] = geom
        return self.nominal_geom_dict[tel_id]




class EventFileLooper(Tool):
    name = "EventFileLooper"
    description = "Loop through the file and apply calibration. Intended as " \
                  "a test that the routines work, and a benchmark of speed."

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
        self.calibrator = None
        self.geometry = None

        self.amp_cut = None
        self.dist_cut = None
        self.tail_cut = None
        self.pix_cut = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.reader = reader_class(**kwargs)

        self.calibrator = CameraCalibrator(origin=self.reader.origin, **kwargs)

        first_event = self.reader.get_event(0)
        self.geometry = Geometry(self.config, self, first_event)

        self.amp_cut = {"LSTCam": 92.7,
                        "NectarCam": 90.6,
                        "FlashCam": 90.6,
                        "CHEC": 29.3}
        self.dist_cut = {"LSTCam": 1.74 * u.deg,
                         "NectarCam": 3. * u.deg,
                         "FlashCam": 3. * u.deg,
                         "CHEC": 3.55 * u.deg}
        self.tail_cut = {"LSTCam": (8, 16),
                         "NectarCam": (7, 14),
                         "FlashCam": (7, 14),
                         "CHEC": (3, 6)}
        self.pix_cut = {"LSTCam": 5,
                         "NectarCam": 4,
                         "FlashCam": 4,
                         "CHEC": 4}

    def start(self):
        # n_events = self.reader.num_events
        source = self.reader.read()
        desc = "Looping through file"
        for event in tqdm(source, desc=desc): #, total=n_events):
            ev = event.count
            self.calibrator.calibrate(event)

            for tel_id in event.r0.tels_with_data:
                geom = self.geometry.get_camera(tel_id)
                nom_geom = self.geometry.get_nominal(tel_id)

                image = event.dl1.tel[tel_id].image[0]

                # Cleaning
                cuts = self.tail_cut[geom.cam_id]
                tc = tailcuts_clean(geom, image, *cuts)
                if not tc.any():
                    # self.log.warning('No image')
                    continue
                # cleaned = np.ma.masked_array(image, mask=~tc)
                cleaned = image * tc

                # Hillas
                try:
                    hillas = hillas_parameters(nom_geom, cleaned)
                except HillasParameterizationError:
                    # self.log.warning('HillasParameterizationError')
                    continue

                # embed()

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                camera = CameraDisplay(nom_geom, ax=ax, image=image, cmap='viridis')
                camera.add_colorbar()
                cen_x = u.Quantity(hillas.cen_x).value
                cen_y = u.Quantity(hillas.cen_y).value
                length = u.Quantity(hillas.length).value
                width = u.Quantity(hillas.width).value

                print(cen_x, cen_y, length, width)

                camera.add_ellipse(centroid=(cen_x, cen_y),
                                   length=length * 2,
                                   width=width * 2,
                                   angle=hillas.psi.rad)

                plt.show()

    def finish(self):
        pass


if __name__ == '__main__':
    exe = EventFileLooper()
    exe.run()
