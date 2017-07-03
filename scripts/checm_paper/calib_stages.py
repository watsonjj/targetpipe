import numpy as np
from matplotlib import pyplot as plt
from traitlets import Dict, List
from ctapipe.core import Tool
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.image.charge_extractors import NeighbourPeakIntegrator
from ctapipe.image.waveform_cleaning import CHECMWaveformCleanerLocal
from ctapipe.image import tailcuts_clean
from ctapipe.image.hillas import hillas_parameters
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.calib.camera.r1 import TargetioR1Calibrator
from targetpipe.plots.official import OfficialPlotter


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
        self.fig.subplots_adjust(right=0.85)

    def create(self, title, units, image, geom, tc,
               hl, hlc='white', hla=0.75, hillas=None):
        camera = CameraDisplay(geom, ax=self.ax,
                               image=image,
                               cmap='viridis')
        camera.add_colorbar()
        camera.colorbar.set_label("Amplitude ({})".format(units), fontsize=20)
        camera.image = image

        # from IPython import embed
        # embed()

        cleaned_image = np.ma.masked_array(image, mask=~tc)
        max_ = cleaned_image.max()
        min_ = np.percentile(image, 0.1)
        camera.set_limits_minmax(min_, max_)
        camera.highlight_pixels(hl, hlc, 1, hla)
        if hillas:
            camera.overlay_moments_update(hillas, color='red', linewidth=2)

        self.ax.set_title(title)
        self.ax.axis('off')
        camera.colorbar.ax.tick_params(labelsize=30)


class WaveformPlotter(OfficialPlotter):
    name = 'WaveformPlotter'

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

    def create(self, waveform, title, units):
        self.ax.plot(waveform)
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (ns)", fontsize=20)
        self.ax.set_ylabel("Amplitude ({})".format(units), fontsize=20)


class CalibStages(Tool):
    name = "CalibStages"
    description = "Plot the different stages in the GCT calibration"

    aliases = Dict(dict())
    classes = List([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reader = None
        self.r1 = None
        self.r1_adc = None
        self.dl0 = None
        self.dl1 = None

        self.p_raw = None
        self.p_calibpedestal = None
        self.p_calibadc = None
        self.p_calibpe = None
        self.p_charge = None
        self.p_cleaned = None
        self.p_hillas = None
        self.p_charge_hillas = None

        self.p_raw_wf = None
        self.p_calibpedestal_wf = None
        self.p_calibadc_wf = None
        self.p_calibpe_wf = None
        self.p_cleaned_wf = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        filepath = '/Volumes/gct-jason/data/170330/onsky-mrk501/Run05477_r0.tio'
        self.reader = TargetioFileReader(input_path=filepath, **kwargs)

        extractor = NeighbourPeakIntegrator(**kwargs)
        cleaner = CHECMWaveformCleanerLocal(**kwargs)

        self.r1 = TargetioR1Calibrator(pedestal_path='/Volumes/gct-jason/data/170330/pedestal/Run05475_ped.tcal',
                                       tf_path='/Volumes/gct-jason/data/170314/tf/Run00001-00050_tf.tcal',
                                       adc2pe_path='/Users/Jason/Software/CHECAnalysis/targetpipe/adc2pe/adc2pe_800gm_c1.tcal',
                                       **kwargs,
                                       )
        self.r1_adc = TargetioR1Calibrator(pedestal_path='/Volumes/gct-jason/data/170330/pedestal/Run05475_ped.tcal',
                                           tf_path='/Volumes/gct-jason/data/170314/tf/Run00001-00050_tf.tcal',
                                           **kwargs,
                                           )
        self.r1_pedestal = TargetioR1Calibrator(pedestal_path='/Volumes/gct-jason/data/170330/pedestal/Run05475_ped.tcal',
                                                **kwargs,
                                                )
        self.dl0 = CameraDL0Reducer(**kwargs)
        self.dl1 = CameraDL1Calibrator(extractor=extractor,
                                       cleaner=cleaner,
                                       **kwargs)

        p_kwargs = kwargs
        p_kwargs['script'] = "checm_paper_calib_stages"
        p_kwargs['figure_name'] = "0_r0_raw"
        self.p_raw = ImagePlotter(**p_kwargs)
        p_kwargs['figure_name'] = "1_r1_calib_pedestal"
        self.p_calibpedestal = ImagePlotter(**p_kwargs)
        p_kwargs['figure_name'] = "2_r1_calib_adc"
        self.p_calibadc = ImagePlotter(**p_kwargs)
        p_kwargs['figure_name'] = "3_r1_calib_pe"
        self.p_calibpe = ImagePlotter(**p_kwargs)
        p_kwargs['figure_name'] = "4_dl1_charge"
        self.p_charge = ImagePlotter(**p_kwargs)
        p_kwargs['figure_name'] = "5_cleaned"
        self.p_cleaned = ImagePlotter(**p_kwargs)
        p_kwargs['figure_name'] = "6_hillas"
        self.p_hillas = ImagePlotter(**p_kwargs)
        p_kwargs['figure_name'] = "7_dl1_charge_hillas"
        self.p_charge_hillas = ImagePlotter(**p_kwargs)

        p_kwargs['figure_name'] = "0_r0_raw_wf"
        self.p_raw_wf = WaveformPlotter(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "1_r1_calib_pedestal_wf"
        self.p_calibpedestal_wf = WaveformPlotter(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "2_r1_calib_adc_wf"
        self.p_calibadc_wf = WaveformPlotter(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "3_r1_calib_pe_wf"
        self.p_calibpe_wf = WaveformPlotter(**p_kwargs, shape='wide')
        p_kwargs['figure_name'] = "4_cleaned_wf"
        self.p_cleaned_wf = WaveformPlotter(**p_kwargs, shape='wide')

    def start(self):

        event_id = 91 #138
        t0 = 40

        event = self.reader.get_event(event_id, True)

        telid = list(event.r0.tels_with_data)[0]
        pos = event.inst.pixel_pos[telid]
        foclen = event.inst.optical_foclen[telid]
        geom = CameraGeometry.guess(*pos, foclen)

        self.r1_adc.calibrate(event)
        r1_adc = np.copy(event.r1.tel[telid].pe_samples)

        self.r1_pedestal.calibrate(event)
        r1_pedestal = np.copy(event.r1.tel[telid].pe_samples)

        self.r1.calibrate(event)
        self.dl0.reduce(event)
        self.dl1.calibrate(event)

        r0 = event.r0.tel[telid].adc_samples
        r1 = event.r1.tel[telid].pe_samples
        dl1 = event.dl1.tel[telid].image[0]
        cleaned = event.dl1.tel[telid].cleaned[0]

        pix = np.argmax(dl1)

        t0_r0 = r0[0, :, t0]
        t0_r1 = r1[0, :, t0]
        t0_r1_adc = r1_adc[0, :, t0]
        t0_r1_pedestal = r1_pedestal[0, :, t0]
        pix_r0 = r0[0, pix, :]
        pix_r1 = r1[0, pix, :]
        pix_r1_adc = r1_adc[0, pix, :]
        pix_r1_pedestal = r1_pedestal[0, pix, :]
        pix_cleaned = cleaned[pix]

        tc = tailcuts_clean(geom, dl1, 20, 10)
        cleaned_dl1 = np.ma.masked_array(dl1, mask=~tc)

        hillas = hillas_parameters(*pos, cleaned_dl1)

        t = "Raw, T=40ns"
        self.p_raw.create(t, "ADC", t0_r0, geom, tc, tc)
        t = "R1 Calibrated Pedestal, T=40ns"
        self.p_calibpedestal.create(t, "ADC", t0_r1_pedestal, geom, tc, tc)
        t = "R1 Calibrated ADC, T=40ns"
        self.p_calibadc.create(t, "ADC", t0_r1_adc, geom, tc, tc)
        t = "R1 Calibrated p.e., T=40ns"
        self.p_calibpe.create(t, "p.e.", t0_r1, geom, tc, tc)
        t = "DL1 Extracted Charge"
        self.p_charge.create(t, "p.e.", dl1, geom, tc, tc)
        t = "Tailcuts Cleaned"
        self.p_cleaned.create(t, "p.e.", cleaned_dl1, geom, tc,
                              np.arange(2048), 'black', 0.2)
        t = "Hillas"
        self.p_hillas.create(t, "p.e.", cleaned_dl1, geom, tc,
                             np.arange(2048), 'black', 0.2, hillas)
        t = "DL1 Extracted Charge"
        self.p_charge_hillas.create(t, "p.e.", dl1, geom, tc, tc, 'white', 0.75, hillas)

        t = "Raw, pix={}".format(pix)
        self.p_raw_wf.create(pix_r0, t, 'ADC')
        t = "R1 Calibrated Pedestal, pix={}".format(pix)
        self.p_calibpedestal_wf.create(pix_r1_pedestal, t, 'ADC')
        t = "R1 Calibrated ADC, pix={}".format(pix)
        self.p_calibadc_wf.create(pix_r1_adc, t, 'ADC')
        t = "R1 Calibrated p.e., pix={}".format(pix)
        self.p_calibpe_wf.create(pix_r1, t, 'p.e.')
        t = "Cleaned Wf, pix={}".format(pix)
        self.p_cleaned_wf.create(pix_cleaned, t, 'p.e.')

    def finish(self):
        self.p_raw.save()
        self.p_calibpedestal.save()
        self.p_calibadc.save()
        self.p_calibpe.save()
        self.p_charge.save()
        self.p_cleaned.save()
        self.p_hillas.save()
        self.p_charge_hillas.save()

        self.p_raw_wf.save()
        self.p_calibpedestal_wf.save()
        self.p_calibadc_wf.save()
        self.p_calibpe_wf.save()
        self.p_cleaned_wf.save()


exe = CalibStages()
exe.run()
