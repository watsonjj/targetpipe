import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy

import astropy.units as u
from traitlets import Dict, List, Unicode, Int, Bool
import numpy as np

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import HessioR1Calibrator

from ctapipe.coordinates import *
from ctapipe.core import Tool
from ctapipe.image import tailcuts_clean, dilate
from ctapipe.instrument import CameraGeometry
from ctapipe.reco.ImPACT import ImPACTReconstructor
from ctapipe.image import hillas_parameters, HillasParameterizationError
from ctapipe.io.hessio import hessio_event_source

from ctapipe.reco.hillas_intersection import HillasIntersection
from ctapipe.reco.energy_regressor import EnergyRegressor
from ctapipe.io.containers import ReconstructedShowerContainer, ReconstructedEnergyContainer

from ctapipe.image import FullIntegrator
import ctapipe.visualization as visualization

from ctapipe.plotting.event_viewer import EventViewer
from astropy.table import Table
class ImPACTReconstruction(Tool):
    """
    
    """
    description = "ImPACTReco"
    name='ctapipe-ImPACT-reco'

    infile = Unicode(help='input simtelarray file').tag(config=True)

    outfile = Unicode(help='output fits table').tag(config=True)

    telescopes = List(Int, None, allow_none=True,
                      help='Telescopes to include from the event file. '
                           'Default = All telescopes').tag(config=True)

    max_events = Int(default_value=1000000000,
                     help="Max number of events to include in analysis").tag(config=True)

    amp_cut = Dict().tag(config=True)
    dist_cut = Dict().tag(config=True)
    tail_cut = Dict().tag(config=True)
    pix_cut = Dict().tag(config=True)

    minimiser = Unicode(default_value="minuit",
                        help='minimiser used for ImPACTReconstruction').tag(config=True)

    aliases = Dict(dict(infile='ImPACTReconstruction.infile',
                        outfile='ImPACTReconstruction.outfile',
                        telescopes='ImPACTReconstruction.telescopes',
                        amp_cut='ImPACTReconstruction.amp_cut',
                        dist_cut='ImPACTReconstruction.dist_cut',
                        tail_cut='ImPACTReconstruction.tail_cut',
                        pix_cut='ImPACTReconstruction.pix_cut',
                        minimiser='ImPACTReconstruction.minimiser',
                        max_events='ImPACTReconstruction.max_events'))

    def setup(self):

        self.geoms = dict()
        if len(self.amp_cut) == 0:
            self.amp_cut = {"LSTCam": 92.7,
                            "NectarCam": 90.6,
                            "FlashCam": 90.6,
                            "CHEC": 29.3}
        if len(self.dist_cut) == 0:
            self.dist_cut = {"LSTCam": 1.74 * u.deg,
                             "NectarCam": 3. * u.deg,
                             "FlashCam": 3. * u.deg,
                             "CHEC": 3.55 * u.deg}
        if len(self.tail_cut) == 0:
            self.tail_cut = {"LSTCam": (8, 16),
                             "NectarCam": (7, 14),
                             "FlashCam": (7, 14),
                             "CHEC": (3, 6)}
        if len(self.pix_cut) == 0:
            self.pix_cut = {"LSTCam": 5,
                             "NectarCam": 4,
                             "FlashCam": 4,
                             "CHEC": 4}

        # Calibrators set to default for now
        self.r1 = HessioR1Calibrator(None, None)
        self.dl0 = CameraDL0Reducer(None, None)
        self.calibrator = CameraDL1Calibrator(None, None,
                                              extractor=FullIntegrator(None, None))

        # If we don't set this just use everything
        if len(self.telescopes) < 2:
            self.telescopes = None

        self.source = hessio_event_source(self.infile, 
                                          allowed_tels=self.telescopes,
                                          max_events=self.max_events)

        self.fit = HillasIntersection()
        self.energy_reco = EnergyRegressor.load("./Aux/{cam_id}.pkl", 
                                                ["CHEC", "LSTCam", "NectarCam"])

        self.ImPACT=ImPACTReconstructor()
        self.viewer = EventViewer(draw_hillas_planes=True)

        self.output = Table(names=['EVENT_ID', 'RECO_ALT', 'RECO_AZ',
                                   'RECO_EN', 'RECO_ALT_HILLAS', 
                                   'RECO_AZ_HILLAS','RECO_EN_HILLAS', 'GOF', 
                                   'SIM_ALT', 'SIM_AZ', 'SIM_EN', 
                                   'NTELS', 'DIST_CORE'],
                            dtype=[np.int64, np.float64, np.float64,
                                   np.float64, np.float64, np.float64, 
                                   np.float64, np.float64, np.float64,
                                   np.float64, np.float64, np.int16, 
                                   np.float64])

    def start(self):

        for event in self.source:
            self.calibrate_event(event)
            self.reconstruct_event(event)

    def finish(self):
        self.output.write(self.outfile)

        return True

    def calibrate_event(self, event):
        """
        Run standard calibrators to get from r0 to dl1
        
        Parameters
        ----------
        event: ctapipe event container

        Returns
        -------
            None
        """
        self.r1.calibrate(event)
        self.dl0.reduce(event)
        self.calibrator.calibrate(event)  # calibrate the events

    def preselect(self, hillas, npix, tel_id):
        """
        Perform pre-selection of telescopes (before reconstruction) based on Hillas
        Parameters
        
        Parameters
        ----------
        hillas: ctapipe Hillas parameter object
        tel_id: int
            Telescope ID number

        Returns
        -------
            bool: Indicate whether telescope passes cuts
        """
        if hillas is None:
            return False

        # Calculate distance of image centroid from camera centre
        dist = np.sqrt(hillas.cen_x * hillas.cen_x + 
                       hillas.cen_y * hillas.cen_y)

        # Cut based on Hillas amplitude and nominal distance
        if hillas.size > self.amp_cut[self.geoms[tel_id].cam_id] and dist < \
                self.dist_cut[self.geoms[tel_id].cam_id] and \
                        hillas.width>0*u.deg and \
                        npix>self.pix_cut[self.geoms[tel_id].cam_id]:
            return True

        return False

    def reconstruct_event(self, event):
        """
        Perform full event reconstruction, including Hillas and ImPACT analysis.
        
        Parameters
        ----------
        event: ctapipe event container

        Returns
        -------
            None
        """
        # store MC pointing direction for the array
        array_pointing = HorizonFrame(alt = event.mcheader.run_array_direction[1]*u.rad,
                                      az = event.mcheader.run_array_direction[0]*u.rad)
        tilted_system = TiltedGroundFrame(pointing_direction=array_pointing)

        image = {}
        pixel_x = {}
        pixel_y = {}
        pixel_area = {}
        tel_type = {}
        tel_x = {}
        tel_y = {}

        hillas = {}
        hillas_nom = {}
        image_pred = {}
        mask_dict = {}
        

        Gate_dict = {}
        Nectar_dict = {}
        LST_dict ={}
        dict_list=[]

        for tel_id in event.dl0.tels_with_data:
            # Get calibrated image (low gain channel only)
            pmt_signal = event.dl1.tel[tel_id].image[0]

            # Create nominal system for the telescope (this should later used 
            #telescope pointing)
            nom_system = NominalFrame(array_direction=array_pointing,
                                      pointing_direction=array_pointing)

            # Create camera system of all pixels
            pix_x, pix_y = event.inst.pixel_pos[tel_id]
            fl = event.inst.optical_foclen[tel_id]
            if tel_id not in self.geoms:
                self.geoms[tel_id] = CameraGeometry.guess(pix_x, pix_y,
                                                          event.inst.optical_foclen[ tel_id],
                                                          apply_derotation=False)


            # Transform the pixels positions into nominal coordinates
            camera_coord = CameraFrame(x=pix_x, y=pix_y,
                                       z=np.zeros(pix_x.shape) * u.m,
                                       focal_length=fl,
                                       rotation= -1* self.geoms[tel_id].cam_rotation)

            nom_coord = camera_coord.transform_to(nom_system)
            tx, ty, tz = event.inst.tel_pos[tel_id]

            # ImPACT reconstruction is performed in the tilted system,
            # so we need to transform tel positions
            grd_tel = GroundFrame(x=tx, y=ty, z=tz)
            tilt_tel = grd_tel.transform_to(tilted_system)

            # Clean image using split level cleaning
            mask = tailcuts_clean(self.geoms[tel_id], pmt_signal,
                                  picture_thresh=self.tail_cut[self.geoms[
                                      tel_id].cam_id][1],
                                  boundary_thresh=self.tail_cut[self.geoms[
                                      tel_id].cam_id][0])

            # Perform Hillas parameterisation
            moments = None
            try:
                moments_cam = hillas_parameters(event.inst.pixel_pos[tel_id][0],
                                            event.inst.pixel_pos[tel_id][1],
                                            pmt_signal*mask)

                moments = hillas_parameters(nom_coord.x, nom_coord.y,pmt_signal*mask)

            except HillasParameterizationError as e:
                print(e)
                continue

            # Make cut based on Hillas parameters
            if self.preselect(moments, np.sum(mask), tel_id):

                # Dialte around edges of image
                for i in range(3):
                    mask = dilate(self.geoms[tel_id], mask)

                # Save everything in dicts for reconstruction later
                pixel_area[tel_id] = self.geoms[tel_id].pix_area/(fl*fl)
                pixel_area[tel_id] *= u.rad*u.rad
                pixel_x[tel_id] = nom_coord.x[mask]
                pixel_y[tel_id] = nom_coord.y[mask]

                tel_x[tel_id] = tilt_tel.x
                tel_y[tel_id] = tilt_tel.y

                tel_type[tel_id] = self.geoms[tel_id].cam_id
                image[tel_id] = pmt_signal[mask]
                image_pred[tel_id] = np.zeros(pmt_signal.shape)

                hillas[tel_id] = moments_cam
                hillas_nom[tel_id] = moments
                mask_dict[tel_id] = mask
                
            
                cam_id = self.geoms[tel_id].cam_id
                print (cam_id)

                mom=[moments.size, tilt_tel, moments.width/u.rad,
                     moments.length/u.rad]
                
                if (cam_id == 'LSTCam'):
                    try:
                        LST_dict[cam_id].append(mom)
                    except:
                        LST_dict.update({'LSTCam':[mom]})
                elif (cam_id =='NectarCam'):
                    try:
                        Nectar_dict['NectarCam'].append(mom)
                    except:
                        Nectar_dict.update({'NectarCam':[mom]})
                elif (cam_id =='CHEC'):
                    try:
                        Gate_dict[cam_id].append(mom)
                    except:
                        Gate_dict.update({'CHEC':[mom]})
                else:
                    print (cam_id +': Type not known')

        #################################################
        # Cut on number of telescopes remaining
        if len(image)>1:
            fit_result = self.fit.predict(hillas_nom, tel_x, 
                                          tel_y, array_pointing)
            
            core_pos_grd = GroundFrame(x=fit_result.core_x,y=fit_result.core_y,
                                       z=0*u.m)
            core_pos_tilt = core_pos_grd.transform_to(tilted_system)

            coredist=np.sqrt(core_pos_grd.x**2+core_pos_grd.y**2)
            
            dict_list=np.array([])
            if len(LST_dict) != 0:
                for i in range (0,len(LST_dict['LSTCam'])):
                    LST_dict['LSTCam'][i][1]= LST_dict['LSTCam'][i][1].separation_3d(core_pos_tilt).to(u.m)/u.m
                dict_list=np.append(dict_list,LST_dict)
            if len(Nectar_dict) != 0:
                for i in range(0,len(Nectar_dict['NectarCam'])):
                    Nectar_dict['NectarCam'][i][1]= Nectar_dict['NectarCam'][i][1].separation_3d(core_pos_tilt).to(u.m) /u.m
                dict_list=np.append(dict_list,Nectar_dict)
            if len(Gate_dict) !=0:
                for i in range(0, len(Gate_dict['CHEC'])):
                    Gate_dict['CHEC'][i][1]= Gate_dict['CHEC'][i][1].separation_3d(core_pos_tilt).to(u.m) /u.m
                dict_list=np.append(dict_list,Gate_dict)

            energy_result = self.energy_reco.predict_by_event(dict_list)

            print('Fit results and Energy results')
            print(energy_result)
            print ('__________________________')

            # Perform ImPACT reconstruction
            self.ImPACT.set_event_properties(image, pixel_x, pixel_y, 
                                             pixel_area, tel_type, tel_x, 
                                             tel_y, array_pointing, hillas_nom)

            energy_seed=ReconstructedEnergyContainer()
            
            energy_seed.energy=np.mean(np.power(10,energy_result['mean'].value)) * u.TeV
            energy_seed.energy_uncert=np.mean(energy_result['std'])
            energy_seed.is_valid = True
            energy_seed.tel_ids= event.dl0.tels_with_data



            ImPACT_shower, ImPACT_energy = self.ImPACT.predict(fit_result, energy_seed)

            print(ImPACT_energy)
            # insert the row into the table
            self.output.add_row((event.dl0.event_id, ImPACT_shower.alt,
                                 ImPACT_shower.az, ImPACT_energy.energy, 
                                 fit_result.alt, fit_result.az, 
                                 np.mean(np.power(10,energy_result['mean'].value)),  
                                 ImPACT_shower.goodness_of_fit,
                                 event.mc.alt, event.mc.az, event.mc.energy, 
                                 len(image),coredist))
            
def main():
    exe = ImPACTReconstruction()
    exe.run()

if __name__ == '__main__':
    main()

